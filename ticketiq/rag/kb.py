"""Knowledge-base loading and chunking.

Chunks are cut on markdown section boundaries first and only split further when
a section exceeds the size budget. Each chunk keeps its heading path prepended,
which materially improves retrieval on short queries because the heading text
carries the topic ("Refund eligibility") that the body may never restate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    heading: str
    text: str

    @property
    def embedding_text(self) -> str:
        # The heading is repeated: it is the most topic-bearing text in the
        # chunk and short queries match it far more reliably than prose.
        return f"{self.heading}. {self.heading}. {self.title}\n{self.text}"


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def chunk_markdown(
    raw: str,
    doc_id: str,
    *,
    chunk_size: int = 480,
    overlap: int = 80,
) -> list[Chunk]:
    meta, body = _parse_frontmatter(raw)
    title = meta.get("title", doc_id.replace("_", " ").title())

    matches = list(_HEADING_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    if not matches:
        sections.append(("", body.strip()))
    else:
        if body[: matches[0].start()].strip():
            sections.append(("", body[: matches[0].start()].strip()))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            sections.append((match.group(2).strip(), body[match.end() : end].strip()))

    chunks: list[Chunk] = []
    for heading, section_text in sections:
        if not section_text:
            continue
        for part in _split_long(section_text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{len(chunks):02d}",
                    title=title,
                    heading=heading or title,
                    text=part.strip(),
                )
            )
    return chunks


def load_kb(kb_dir: Path, *, chunk_size: int = 480, overlap: int = 80) -> list[Chunk]:
    """Load and chunk every markdown document in ``kb_dir``."""
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"knowledge base directory not found: {kb_dir}")
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_markdown(raw, path.stem, chunk_size=chunk_size, overlap=overlap)
        )
    if not chunks:
        raise ValueError(f"no markdown documents found in {kb_dir}")
    return chunks
