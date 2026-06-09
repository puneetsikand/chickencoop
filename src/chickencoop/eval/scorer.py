"""
Score a single (proposed_text, ground_truth) extraction pair.

The scorer does not embed or semantically compare prose — it checks structural
compliance (schema, required sections), type-level correctness, and whether
the model honoured the existing-IDs constraint. Semantic scoring (headline
similarity, body alignment) is a later layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import frontmatter


@dataclass
class ScoreCard:
    nugget_id: str
    proposed_id: str | None = None
    proposed_type: str | None = None
    schema_valid: bool = False
    type_match: bool | None = None
    related_validity: float = 0.0   # fraction of proposed related IDs that exist in corpus
    related_overlap: float = 0.0    # Jaccard of proposed vs ground-truth related sets
    has_headline: bool = False
    has_why_matters: bool = False
    has_how_to_apply: bool = False
    has_cross_references: bool = False
    refused: bool = False
    parse_error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.schema_valid
            and bool(self.type_match)
            and self.related_validity == 1.0
            and self.has_headline
        )


_FENCE_RE = re.compile(r"^```(?:markdown)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def parse_proposed(text: str | None) -> tuple[dict[str, Any] | None, str, str | None]:
    """Parse a proposed nugget into (metadata, body, error)."""
    if text is None:
        return None, "", "model returned NO_NUGGET"

    cleaned = text.strip()
    fence = _FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1)

    try:
        post = frontmatter.loads(cleaned)
        return dict(post.metadata), post.content, None
    except Exception as e:
        return None, "", f"frontmatter parse failed: {type(e).__name__}: {e}"


def score_extraction(
    *,
    proposed_text: str | None,
    ground_truth_id: str,
    ground_truth_type: str,
    ground_truth_related: list[str],
    corpus_ids: set[str],
) -> ScoreCard:
    card = ScoreCard(nugget_id=ground_truth_id)

    if proposed_text is None:
        card.refused = True
        return card

    meta, body, err = parse_proposed(proposed_text)
    if err or meta is None:
        card.parse_error = err
        return card

    card.schema_valid = True
    card.proposed_id = meta.get("id")
    card.proposed_type = meta.get("type")
    card.type_match = card.proposed_type == ground_truth_type

    proposed_related = list(meta.get("related") or [])
    if proposed_related:
        valid = sum(1 for r in proposed_related if r in corpus_ids)
        card.related_validity = valid / len(proposed_related)
        a, b = set(proposed_related), set(ground_truth_related)
        card.related_overlap = len(a & b) / len(a | b) if (a or b) else 1.0
    else:
        # No proposed related — vacuously valid; overlap is 1 only if GT also empty
        card.related_validity = 1.0
        card.related_overlap = 1.0 if not ground_truth_related else 0.0

    card.has_headline = body.lstrip().startswith("# ")
    card.has_why_matters = "## Why it matters" in body or "## Why this matters" in body
    card.has_how_to_apply = "## How to apply" in body
    card.has_cross_references = (
        "## Cross-references" in body or "## Cross-refs" in body
    )

    if card.proposed_id == ground_truth_id:
        card.notes.append("proposed id collides with ground-truth id (model used held-out)")

    invalid_related = [r for r in proposed_related if r not in corpus_ids]
    if invalid_related:
        card.notes.append(f"invented related IDs: {invalid_related}")

    return card
