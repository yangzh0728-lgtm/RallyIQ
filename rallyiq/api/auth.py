"""Basic email + password accounts with server-side sessions.

Off by default (local single-user mode: every request is anonymous and sees every video).
Online, set ``RALLYIQ_AUTH=on``; then every ``/api`` route except health, config and auth
needs a session, and each user only sees their own videos.

Env:
    RALLYIQ_AUTH            off | on                         (default off)
    RALLYIQ_SIGNUP          invite | open | closed           (default invite)
    RALLYIQ_INVITE_CODE     code required when RALLYIQ_SIGNUP=invite
    RALLYIQ_COOKIE_SECURE   1 to mark the session cookie Secure (set behind HTTPS)
    RALLYIQ_SESSION_DAYS    session lifetime (default 30)

The session token travels in an HttpOnly cookie, so ``<video src>`` and download links work
without extra headers; API clients can send ``Authorization: Bearer <token>`` instead.

Create an account from the shell (e.g. the first admin when signup is closed)::

    python -m rallyiq.api.auth create-user you@example.com
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import os
import re
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..store import Store

COOKIE = "rallyiq_session"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8


# ------------------------------------------------------------------ passwords

def hash_password(password: str, *, n: int = 2**14, r: int = 8, p: int = 1) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${n}${r}${p}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, dk = stored.split("$")
        if algo != "scrypt":
            return False
        want = base64.b64decode(dk)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(want))
        return hmac.compare_digest(got, want)
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --------------------------------------------------------------------- config

@dataclass
class AuthConfig:
    enabled: bool = False
    signup: Literal["invite", "open", "closed"] = "invite"
    invite_code: str | None = None
    cookie_secure: bool = False
    session_days: int = 30

    @classmethod
    def from_env(cls) -> AuthConfig:
        signup = os.environ.get("RALLYIQ_SIGNUP", "invite").strip().lower()
        if signup not in ("invite", "open", "closed"):
            raise ValueError(f"RALLYIQ_SIGNUP must be invite, open or closed, not {signup!r}")
        return cls(
            enabled=os.environ.get("RALLYIQ_AUTH", "off").strip().lower() in ("1", "on", "true", "yes"),
            signup=signup,  # type: ignore[arg-type]
            invite_code=os.environ.get("RALLYIQ_INVITE_CODE") or None,
            cookie_secure=os.environ.get("RALLYIQ_COOKIE_SECURE", "0") in ("1", "true", "yes", "on"),
            session_days=int(os.environ.get("RALLYIQ_SESSION_DAYS", "30")),
        )


@dataclass
class User:
    user_id: str
    email: str
    is_admin: bool = False


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class SignupIn(Credentials):
    invite_code: str | None = None


class UserOut(BaseModel):
    user_id: str
    email: str
    is_admin: bool = False


class RateLimiter:
    """Sliding-window limit per key (per process; enough to slow down password guessing)."""

    def __init__(self, limit: int, window_s: float):
        self.limit, self.window_s = limit, window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        t = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] < t - self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(t)
            return True


def _row_user(r) -> User:
    return User(user_id=r["user_id"], email=r["email"], is_admin=bool(r["is_admin"]))


def normalize_email(email: str) -> str:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, "邮箱格式不正确")
    return email


def build_auth(store: Store, cfg: AuthConfig):
    """Returns ``(router, current_user)``; ``current_user`` is a FastAPI dependency."""

    router = APIRouter(prefix="/api/auth", tags=["auth"])
    limiter = RateLimiter(limit=10, window_s=600)

    def session_token(request: Request) -> str | None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip() or None
        return request.cookies.get(COOKIE)

    def lookup(request: Request) -> User | None:
        tok = session_token(request)
        if not tok:
            return None
        r = store.session_user(token_hash(tok))
        return _row_user(r) if r else None

    def current_user(request: Request) -> User | None:
        if not cfg.enabled:
            return None
        user = lookup(request)
        if user is None:
            raise HTTPException(401, "请先登录", headers={"WWW-Authenticate": "Bearer"})
        return user

    def start_session(response: Response, user_id: str) -> str:
        tok = secrets.token_urlsafe(32)
        ttl = timedelta(days=cfg.session_days)
        store.create_session(token_hash(tok), user_id, ttl)
        response.set_cookie(COOKIE, tok, max_age=int(ttl.total_seconds()), httponly=True, samesite="lax",
                            secure=cfg.cookie_secure, path="/")
        return tok

    def require_enabled() -> None:
        if not cfg.enabled:
            raise HTTPException(404, "accounts are disabled (RALLYIQ_AUTH=off)")

    def client_key(request: Request, email: str) -> str:
        ip = request.client.host if request.client else "?"  # uvicorn --proxy-headers resolves X-Forwarded-For
        return f"{ip}|{email}"

    @router.post("/signup", response_model=UserOut, status_code=201)
    def signup(body: SignupIn, request: Request, response: Response) -> UserOut:
        require_enabled()
        if cfg.signup == "closed":
            raise HTTPException(403, "暂不开放注册")
        if cfg.signup == "invite":
            if not cfg.invite_code or not hmac.compare_digest((body.invite_code or "").strip(), cfg.invite_code):
                raise HTTPException(403, "邀请码不正确")
        email = normalize_email(body.email)
        if not limiter.hit(client_key(request, "signup")):
            raise HTTPException(429, "尝试次数过多，请稍后再试")
        if len(body.password) < MIN_PASSWORD:
            raise HTTPException(422, f"密码至少 {MIN_PASSWORD} 位")
        if store.get_user_by_email(email):
            raise HTTPException(409, "该邮箱已注册")
        is_admin = store.count_users() == 0
        user_id = store.create_user(email, hash_password(body.password), is_admin=is_admin)
        start_session(response, user_id)
        return UserOut(user_id=user_id, email=email, is_admin=is_admin)

    @router.post("/login", response_model=UserOut)
    def login(body: Credentials, request: Request, response: Response) -> UserOut:
        require_enabled()
        email = body.email.strip().lower()
        if not limiter.hit(client_key(request, email)):
            raise HTTPException(429, "尝试次数过多，请稍后再试")
        r = store.get_user_by_email(email)
        if r is None or not verify_password(body.password, r["password_hash"]):
            raise HTTPException(401, "邮箱或密码不正确")
        start_session(response, r["user_id"])
        u = _row_user(r)
        return UserOut(user_id=u.user_id, email=u.email, is_admin=u.is_admin)

    @router.post("/logout", status_code=204)
    def logout(request: Request, response: Response) -> Response:
        tok = session_token(request)
        if tok:
            store.delete_session(token_hash(tok))
        response.status_code = 204
        response.delete_cookie(COOKIE, path="/")
        return response

    @router.get("/me", response_model=UserOut)
    def me(request: Request) -> UserOut:
        require_enabled()
        u = lookup(request)
        if u is None:
            raise HTTPException(401, "请先登录")
        return UserOut(user_id=u.user_id, email=u.email, is_admin=u.is_admin)

    return router, current_user


# ------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rallyiq.api.auth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-user", help="create an account (password read from stdin / prompt)")
    c.add_argument("email")
    c.add_argument("--admin", action="store_true")
    args = ap.parse_args(argv)
    store = Store()
    email = args.email.strip().lower()
    if not EMAIL_RE.match(email):
        print("invalid email", file=sys.stderr)
        return 2
    if store.get_user_by_email(email):
        print("user exists", file=sys.stderr)
        return 1
    pw = getpass.getpass("password: ") if sys.stdin.isatty() else sys.stdin.readline().rstrip("\n")
    if len(pw) < MIN_PASSWORD:
        print(f"password must be at least {MIN_PASSWORD} characters", file=sys.stderr)
        return 2
    print(store.create_user(email, hash_password(pw), is_admin=args.admin))
    return 0


if __name__ == "__main__":
    sys.exit(main())
