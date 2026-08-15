"""
indexing.py
-----------
Builds and persists the two indexes that power hybrid search:

1. Dense index (FAISS)   - semantic similarity via sentence embeddings.
                            Good at matching meaning/paraphrase.
2. Sparse index (BM25)   - classic keyword/term-frequency search.
                            Good at matching exact terminology, model
                            names, acronyms, equation symbols, etc.
                            that embeddings can blur together.

Both are cheap to build locally and need no external API.
"""

import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
from ingest import Chunk


def _tokenize(text: str) -> List[str]:
    """Simple lowercase whitespace/punctuation tokenizer for BM25.
    Deliberately simple (no external NLTK data download needed)."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridIndex:
    """Owns the embedding model, FAISS index, and BM25 index together."""

    def __init__(self, dense_model_name: str = config.DENSE_MODEL_NAME):
        print(f"Loading dense embedding model '{dense_model_name}' ...")
        self.embedder = SentenceTransformer(dense_model_name)
        self.chunks: List[Chunk] = []
        self.faiss_index: faiss.Index = None
        self.bm25: BM25Okapi = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        print(f"Embedding {len(texts)} chunks for dense index ...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # so inner product == cosine similarity
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        # Inner product on normalized vectors = cosine similarity search.
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        print("Building BM25 sparse index ...")
        tokenized_corpus = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=config.BM25_K1, b=config.BM25_B)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self,
             faiss_path: Path = config.FAISS_INDEX_PATH,
             bm25_path: Path = config.BM25_INDEX_PATH,
             chunks_path: Path = config.CHUNKS_PATH) -> None:
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(faiss_path))
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        from ingest import save_chunks
        save_chunks(self.chunks, chunks_path)
        print(f"Saved dense index -> {faiss_path}")
        print(f"Saved BM25 index  -> {bm25_path}")
        print(f"Saved chunks      -> {chunks_path}")

    @classmethod
    def load(cls,
             faiss_path: Path = config.FAISS_INDEX_PATH,
             bm25_path: Path = config.BM25_INDEX_PATH,
             chunks_path: Path = config.CHUNKS_PATH,
             dense_model_name: str = config.DENSE_MODEL_NAME) -> "HybridIndex":
        from ingest import load_chunks
        obj = cls(dense_model_name)
        obj.faiss_index = faiss.read_index(str(faiss_path))
        with open(bm25_path, "rb") as f:
            obj.bm25 = pickle.load(f)
        obj.chunks = load_chunks(chunks_path)
        return obj

    # ------------------------------------------------------------------
    # Raw retrieval primitives (used by retrieval.py)
    # ------------------------------------------------------------------
    def dense_search(self, query: str, top_k: int):
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idxs = self.faiss_index.search(q_emb, top_k)
        return list(zip(idxs[0].tolist(), scores[0].tolist()))

    def bm25_search(self, query: str, top_k: int):
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]


def build_and_save_index(directory: Path = config.PAPERS_DIR) -> HybridIndex:
    from ingest import ingest_directory
    chunks = ingest_directory(directory)
    index = HybridIndex()
    index.build(chunks)
    index.save()
    return index


if __name__ == "__main__":
    build_and_save_index()
