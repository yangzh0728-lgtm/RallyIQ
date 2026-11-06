"""``python -m rallyiq.review export --out hard_examples/``: dump hard examples and the per-condition summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..store import Store
from .hard_examples import collect, summarize


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rallyiq.review")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write hard_examples.jsonl + summary.json")
    ex.add_argument("--data", type=Path, help="data dir (default: RALLYIQ_DATA or ./data)")
    ex.add_argument("--out", type=Path, default=Path("hard_examples"))
    args = ap.parse_args(argv)

    store = Store(args.data)
    examples, videos = collect(store)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "hard_examples.jsonl").open("w") as f:
        for e in examples:
            f.write(e.model_dump_json() + "\n")
    summary = summarize(examples, videos)
    (args.out / "summary.json").write_text(summary.model_dump_json(indent=1))
    print(f"{summary.total} hard examples ({summary.errors} model errors) from {summary.videos} videos -> {args.out}")
    for g in summary.weakest:
        print(f"  weakest: {g.condition}={g.value}: hit error rate {g.hit_error_rate:.1%} over {g.model_shots} shots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
