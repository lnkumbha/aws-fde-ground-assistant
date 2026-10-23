from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    section: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def _clean_markdown(text: str) -> str:
    # Keep heading text; remove common Markdown syntax to improve lexical retrieval.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[`*_>#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_words(text: str, chunk_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be non-negative and smaller than chunk_words")

    chunks = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def load_markdown_documents(
    directory: Path, chunk_words: int = 120, overlap: int = 25
) -> list[Chunk]:
    """Load Markdown files and preserve the most recent heading as section metadata."""
    if not directory.exists():
        return []

    chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        current_section = "Introduction"
        section_lines: list[str] = []

        def flush_section() -> None:
            cleaned = _clean_markdown(" ".join(section_lines))
            for piece in _split_words(cleaned, chunk_words, overlap):
                chunks.append(
                    Chunk(
                        source=str(path.relative_to(directory)),
                        section=current_section,
                        text=piece,
                    )
                )

        for line in raw.splitlines():
            heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
            if heading:
                flush_section()
                current_section = heading.group(1).strip()
                section_lines = []
            else:
                section_lines.append(line)
        flush_section()

    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if chunks:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(self, question: str, limit: int = 3) -> list[SearchHit]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:limit]
        return [
            SearchHit(self.chunks[i], float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0
        ]
