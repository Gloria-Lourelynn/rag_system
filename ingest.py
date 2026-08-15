"""
ingest.py
---------
Turns raw research papers (PDF or plain text) into overlapping,
metadata-tagged chunks that the indexer can embed and search.

Each chunk keeps track of exactly which paper and which page it came
from. That provenance is what makes verified citations possible later:
we never let the model cite a passage that we can't point back to.
"""

import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pdfplumber

import config


@dataclass
class Chunk:
    chunk_id: str        # unique id, e.g. "attention_is_all_you_need_p3_c1"
    text: str             # the chunk's text
    source: str           # filename of the paper
    page: int              # 1-indexed page number (best-effort for .txt)
    chunk_index: int       # position of this chunk within the page


def _split_into_words(text: str) -> List[str]:
    return text.split()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding-window word chunking with overlap.

    Word-based (not character-based) chunking keeps chunks roughly
    uniform in "reading length" regardless of formatting quirks in the
    source PDF.
    """
    words = _split_into_words(text)
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def _clean_text(text: str) -> str:
    """Light cleanup of PDF text extraction artifacts."""
    text = re.sub(r"-\n", "", text)          # de-hyphenate line-wrapped words
    text = re.sub(r"\s+", " ", text)          # collapse whitespace/newlines
    return text.strip()


def load_pdf(path: Path) -> List[Chunk]:
    """Extract text page-by-page from a PDF and chunk each page."""
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            text = _clean_text(raw)
            if not text:
                continue
            pieces = _chunk_text(text, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
            for i, piece in enumerate(pieces):
                chunks.append(Chunk(
                    chunk_id=f"{path.stem}_p{page_num}_c{i}",
                    text=piece,
                    source=path.name,
                    page=page_num,
                    chunk_index=i,
                ))
    return chunks


def load_txt(path: Path) -> List[Chunk]:
    """Chunk a plain-text paper. Treated as a single 'page' (page=1)
    since .txt files carry no native pagination."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _clean_text(raw)
    pieces = _chunk_text(text, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    return [
        Chunk(
            chunk_id=f"{path.stem}_p1_c{i}",
            text=piece,
            source=path.name,
            page=1,
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
    ]


def ingest_directory(directory: Path = config.PAPERS_DIR) -> List[Chunk]:
    """Ingest every .pdf/.txt file in a directory into a flat chunk list."""
    directory = Path(directory)
    all_chunks: List[Chunk] = []
    files = sorted(list(directory.glob("*.pdf")) + list(directory.glob("*.txt")))

    if not files:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in {directory}. "
            f"Add your research papers there first."
        )

    for path in files:
        if path.suffix.lower() == ".pdf":
            file_chunks = load_pdf(path)
        else:
            file_chunks = load_txt(path)
        print(f"  ingested {path.name}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    return all_chunks


def save_chunks(chunks: List[Chunk], path: Path = config.CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: Path = config.CHUNKS_PATH) -> List[Chunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks


if __name__ == "__main__":
    chunks = ingest_directory()
    save_chunks(chunks)
    print(f"Total chunks: {len(chunks)} -> saved to {config.CHUNKS_PATH}")
