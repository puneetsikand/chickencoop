"""
Reads the nugget corpus from a local directory.
Returns only status: current nuggets — superseded nuggets are invisible to extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import frontmatter


@dataclass
class Nugget:
    id: str
    type: str
    status: str
    created: str
    body: str
    sources: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    sotu_snapshot: bool = False
    path: Path = field(default_factory=Path)


def load_corpus(nugget_dir: str | Path) -> list[Nugget]:
    """Load all status:current nuggets from a directory of markdown files."""
    nugget_dir = Path(nugget_dir)
    nuggets: list[Nugget] = []
    for md_file in sorted(nugget_dir.glob("*.md")):
        if md_file.name in ("README.md", "INDEX.md"):
            continue
        try:
            post = frontmatter.load(md_file)
        except Exception:
            continue
        if post.get("status") != "current":
            continue
        nuggets.append(
            Nugget(
                id=post.get("id", md_file.stem),
                type=post.get("type", ""),
                status=post.get("status", ""),
                created=str(post.get("created", "")),
                body=post.content,
                sources=post.get("sources") or [],
                related=post.get("related") or [],
                supersedes=post.get("supersedes") or [],
                sotu_snapshot=bool(post.get("sotu_snapshot", False)),
                path=md_file,
            )
        )
    return nuggets


def current_ids(nugget_dir: str | Path) -> list[str]:
    """Return sorted list of IDs for all current nuggets."""
    return [n.id for n in load_corpus(nugget_dir)]


def iter_passages(nugget_dir: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (id, full_text_with_frontmatter) for embedding index builds."""
    nugget_dir = Path(nugget_dir)
    for md_file in sorted(nugget_dir.glob("*.md")):
        if md_file.name in ("README.md", "INDEX.md"):
            continue
        try:
            post = frontmatter.load(md_file)
        except Exception:
            continue
        if post.get("status") != "current":
            continue
        nid = post.get("id", md_file.stem)
        yield nid, md_file.read_text()
