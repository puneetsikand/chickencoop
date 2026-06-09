"""
chickencoop CLI — dev-time entry point.

Usage:
  chickencoop extract --passage <file> --corpus <dir> [--ref <source-ref>]
  chickencoop ids --corpus <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chickencoop.corpus.loader import current_ids
from chickencoop.extraction.runner import extract_nugget
from chickencoop.eval.harness import run_eval


def cmd_extract(args: argparse.Namespace) -> None:
    passage_text = Path(args.passage).read_text()
    ids = current_ids(args.corpus)
    nugget = extract_nugget(
        source_passage=passage_text,
        source_ref=args.ref or args.passage,
        existing_ids=ids,
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        think_log=Path(args.think_log) if args.think_log else None,
    )
    if nugget is None:
        sys.exit(0)
    print(nugget)


def cmd_ids(args: argparse.Namespace) -> None:
    for nid in current_ids(args.corpus):
        print(nid)


def cmd_eval(args: argparse.Namespace) -> None:
    run_eval(
        corpus_dir=Path(args.corpus),
        mode=args.mode,
        limit=args.limit,
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        out_path=Path(args.out) if args.out else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="chickencoop")
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="Extract a nugget from a passage")
    ex.add_argument("--passage", required=True, help="Path to the source text file")
    ex.add_argument("--corpus", required=True, help="Path to the nuggets directory")
    ex.add_argument("--ref", default=None, help="Source reference label (session, thread, etc.)")
    ex.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ex.add_argument("--model", default="deepseek-r1-distill-qwen-32b")
    ex.add_argument("--think-log", default=None, help="File to append <think> blocks for calibration")
    ex.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds (default 600)")
    ex.set_defaults(func=cmd_extract)

    ids_cmd = sub.add_parser("ids", help="List current nugget IDs in the corpus")
    ids_cmd.add_argument("--corpus", required=True, help="Path to the nuggets directory")
    ids_cmd.set_defaults(func=cmd_ids)

    ev = sub.add_parser("eval", help="Run extraction eval against the corpus")
    ev.add_argument("--corpus", required=True, help="Path to the nuggets directory")
    ev.add_argument("--mode", default="self", choices=["self"],
                    help="self: strip each nugget's headline and ask model to reconstruct")
    ev.add_argument("--limit", type=int, default=None, help="Stop after N nuggets")
    ev.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ev.add_argument("--model", default="deepseek-r1-distill-qwen-32b")
    ev.add_argument("--timeout", type=int, default=600)
    ev.add_argument("--out", default=None, help="JSON report output path")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)
