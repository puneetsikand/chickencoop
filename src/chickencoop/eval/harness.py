"""
Eval harness: loop over the corpus, run extraction on each nugget's body-minus-headline,
score the proposed output, write a JSON report.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from chickencoop.corpus.loader import load_corpus
from chickencoop.extraction.runner import extract_nugget
from chickencoop.eval.scorer import ScoreCard, score_extraction


def strip_headline(body: str) -> str:
    """Remove the first H1 line so the model has to reconstruct the thesis."""
    text = body.lstrip()
    if text.startswith("# "):
        nl = text.find("\n")
        return text[nl + 1 :].lstrip() if nl >= 0 else ""
    return body


def make_passage(nugget, mode: str) -> str:
    if mode == "self":
        return strip_headline(nugget.body).strip()
    raise ValueError(f"unknown eval mode: {mode}")


def run_eval(
    *,
    corpus_dir: Path,
    mode: str = "self",
    limit: int | None = None,
    endpoint: str = "http://localhost:8000/v1/chat/completions",
    model: str = "deepseek-r1-distill-qwen-32b",
    timeout: int = 600,
    out_path: Path | None = None,
    today: str | None = None,
) -> list[ScoreCard]:
    nuggets = load_corpus(corpus_dir)
    corpus_ids = {n.id for n in nuggets}
    today_str = today or date.today().isoformat()

    if limit:
        nuggets = nuggets[:limit]

    results: list[ScoreCard] = []
    total = len(nuggets)
    for i, nugget in enumerate(nuggets, 1):
        passage = make_passage(nugget, mode)
        held_out = sorted(corpus_ids - {nugget.id})

        print(f"[{i}/{total}] {nugget.id} ({nugget.type})", flush=True)

        try:
            proposed = extract_nugget(
                source_passage=passage,
                source_ref=f"eval/{nugget.id}",
                existing_ids=held_out,
                endpoint=endpoint,
                model=model,
                timeout=timeout,
                today=today_str,
            )
        except Exception as e:
            card = ScoreCard(nugget_id=nugget.id, parse_error=f"extraction failed: {e}")
            results.append(card)
            print(f"  ERROR: {e}", flush=True)
            continue

        card = score_extraction(
            proposed_text=proposed,
            ground_truth_id=nugget.id,
            ground_truth_type=nugget.type,
            ground_truth_related=nugget.related,
            corpus_ids=corpus_ids,
        )
        results.append(card)
        _print_card(card)

    summary = summarize(results)
    print("\n--- SUMMARY ---", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {_fmt(v)}", flush=True)

    if out_path:
        write_report(results, summary, out_path)
        print(f"\nReport: {out_path}", flush=True)

    return results


def _print_card(card: ScoreCard) -> None:
    if card.refused:
        print("  NO_NUGGET", flush=True)
        return
    if card.parse_error:
        print(f"  PARSE_ERROR: {card.parse_error}", flush=True)
        return
    tag = "PASS" if card.passed else "FAIL"
    type_mark = "✓" if card.type_match else "✗"
    print(
        f"  {tag} | type={card.proposed_type}{type_mark}"
        f" | related_valid={card.related_validity:.0%}"
        f" | related_overlap={card.related_overlap:.0%}",
        flush=True,
    )
    for note in card.notes:
        print(f"    note: {note}", flush=True)


def summarize(results: list[ScoreCard]) -> dict:
    n = len(results)
    if n == 0:
        return {"count": 0}
    scored = [r for r in results if not r.refused and not r.parse_error]
    s = len(scored) or 1
    return {
        "count": n,
        "refused": sum(1 for r in results if r.refused) / n,
        "parse_errors": sum(1 for r in results if r.parse_error) / n,
        "schema_valid": sum(1 for r in results if r.schema_valid) / n,
        "type_match": sum(1 for r in scored if r.type_match) / s,
        "headline_present": sum(1 for r in scored if r.has_headline) / s,
        "why_matters_present": sum(1 for r in scored if r.has_why_matters) / s,
        "how_to_apply_present": sum(1 for r in scored if r.has_how_to_apply) / s,
        "cross_refs_present": sum(1 for r in scored if r.has_cross_references) / s,
        "mean_related_validity": sum(r.related_validity for r in scored) / s,
        "mean_related_overlap": sum(r.related_overlap for r in scored) / s,
        "pass_rate": sum(1 for r in results if r.passed) / n,
    }


def write_report(results: list[ScoreCard], summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(data, indent=2))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2%}"
    return v
