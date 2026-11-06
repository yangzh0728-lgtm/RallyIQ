"""Object storage for videos and stage artifacts.

``RALLYIQ_STORAGE`` selects the backend::

    (unset)                -> local files under the data dir (keys are relative paths)
    s3://bucket[/prefix]   -> any S3-compatible store (AWS S3, Cloudflare R2, MinIO)

S3 settings: ``RALLYIQ_S3_ENDPOINT`` (R2 / MinIO URL; omit for AWS), ``RALLYIQ_S3_REGION``
(default ``auto``), credentials from the usual ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``,
``RALLYIQ_S3_PUBLIC_ENDPOINT`` (optional: host the browser should use in presigned URLs,
e.g. ``http://localhost:9000`` when the API reaches MinIO as ``http://minio:9000``).

Keys look like ``videos/<video_id>/source.mp4`` and ``videos/<video_id>/artifacts/<name>.json``,
which is exactly the local data-dir layout, so a local data dir can be synced to a bucket as-is.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    kind: str

    def put_file(self, key: str, src: Path, content_type: str | None = None) -> None: ...
    def get_file(self, key: str, dest: Path) -> None: ...
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None: ...
    def get_bytes(self, key: str) -> bytes | None: ...
    def exists(self, key: str) -> bool: ...
    def size(self, key: str) -> int | None: ...
    def stamp(self, key: str) -> list[int] | None: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def local_path(self, key: str) -> Path | None: ...
    def presign_get(self, key: str, expires_s: int = 3600, filename: str | None = None) -> str | None: ...
    def presign_put(self, key: str, content_type: str, expires_s: int = 3600) -> str | None: ...


class LocalBlobStore:
    """Keys are paths under ``root``. ``local_path`` lets callers stream files directly."""

    kind = "local"

    def __init__(self, root: Path):
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"bad key {key!r}")
        return p

    def put_file(self, key: str, src: Path, content_type: str | None = None) -> None:
        dest = self._p(key)
        if Path(src).resolve() == dest:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(dest)

    def get_file(self, key: str, dest: Path) -> None:
        src = self._p(key)
        if src != Path(dest).resolve():
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        dest = self._p(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)

    def get_bytes(self, key: str) -> bytes | None:
        p = self._p(key)
        return p.read_bytes() if p.exists() else None

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def size(self, key: str) -> int | None:
        p = self._p(key)
        return p.stat().st_size if p.exists() else None

    def stamp(self, key: str) -> list[int] | None:
        """Changes whenever the object is rewritten: ``[mtime_ns, size]``, or None if missing."""
        try:
            st = self._p(key).stat()
        except OSError:
            return None
        return [st.st_mtime_ns, st.st_size]

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        p = self._p(prefix.rstrip("/"))
        if p.is_dir():
            shutil.rmtree(p)

    def local_path(self, key: str) -> Path | None:
        return self._p(key)

    def presign_get(self, key: str, expires_s: int = 3600, filename: str | None = None) -> str | None:
        return None

    def presign_put(self, key: str, content_type: str, expires_s: int = 3600) -> str | None:
        return None


class S3BlobStore:
    kind = "s3"

    def __init__(self, bucket: str, prefix: str = "", endpoint: str | None = None, region: str | None = None,
                 public_endpoint: str | None = None):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:  # pragma: no cover - depends on install
            raise RuntimeError("S3 storage needs boto3: pip install '.[server]'") from e
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 5})
        kw = dict(region_name=region or "auto", config=cfg)
        self.client = boto3.client("s3", endpoint_url=endpoint, **kw)
        # Presigned URLs are signed for the host the *browser* will use.
        self.public_client = (
            boto3.client("s3", endpoint_url=public_endpoint, **kw) if public_endpoint else self.client
        )

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, key: str, src: Path, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else None
        self.client.upload_file(str(src), self.bucket, self._k(key), ExtraArgs=extra)

    def get_file(self, key: str, dest: Path) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(dest).with_name(Path(dest).name + ".part")
        self.client.download_file(self.bucket, self._k(key), str(tmp))
        tmp.replace(dest)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._k(key), Body=data,
                               **({"ContentType": content_type} if content_type else {}))

    def _missing(self, e: Exception) -> bool:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        return code in {"404", "NoSuchKey", "NotFound"}

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()
        except Exception as e:  # noqa: BLE001
            if self._missing(e):
                return None
            raise

    def size(self, key: str) -> int | None:
        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=self._k(key))["ContentLength"])
        except Exception as e:  # noqa: BLE001
            if self._missing(e):
                return None
            raise

    def exists(self, key: str) -> bool:
        return self.size(key) is not None

    def stamp(self, key: str) -> list[int] | None:
        try:
            h = self.client.head_object(Bucket=self.bucket, Key=self._k(key))
        except Exception as e:  # noqa: BLE001
            if self._missing(e):
                return None
            raise
        return [int(h["LastModified"].timestamp() * 1e9), int(h["ContentLength"])]

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._k(key))

    def delete_prefix(self, prefix: str) -> None:
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self._k(prefix))
        for page in pages:
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})

    def local_path(self, key: str) -> Path | None:
        return None

    def presign_get(self, key: str, expires_s: int = 3600, filename: str | None = None) -> str | None:
        params = {"Bucket": self.bucket, "Key": self._k(key)}
        if filename:
            params["ResponseContentDisposition"] = f'inline; filename="{filename}"'
        return self.public_client.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_s)

    def presign_put(self, key: str, content_type: str, expires_s: int = 3600) -> str | None:
        return self.public_client.generate_presigned_url(
            "put_object", Params={"Bucket": self.bucket, "Key": self._k(key), "ContentType": content_type},
            ExpiresIn=expires_s,
        )


def blob_store_from_env(data_dir: Path) -> BlobStore:
    url = os.environ.get("RALLYIQ_STORAGE", "").strip()
    if not url or url == "local":
        return LocalBlobStore(data_dir)
    if url.startswith("s3://"):
        bucket, _, prefix = url[5:].partition("/")
        return S3BlobStore(
            bucket, prefix,
            endpoint=os.environ.get("RALLYIQ_S3_ENDPOINT") or None,
            region=os.environ.get("RALLYIQ_S3_REGION") or None,
            public_endpoint=os.environ.get("RALLYIQ_S3_PUBLIC_ENDPOINT") or None,
        )
    raise ValueError(f"unsupported RALLYIQ_STORAGE {url!r}")
