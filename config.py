"""
config.py
---------
Central place for every tunable setting in the system.
Change values here instead of hunting through the codebase.

NOTE ON MODELS: every model below is downloaded once from Hugging Face
(free, no API key) and then cached locally in ~/.cache/huggingface.
After the first run, everything works fully offline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "data" / "papers"      # drop your PDFs/.txt papers here
STORAGE_DIR = BASE_DIR / "storage"             # persisted indexes live here

FAISS_INDEX_PATH = STORAGE_DIR / "dense.index"
BM25_INDEX_PATH = STORAGE_DIR / "bm25.pkl"
CHUNKS_PATH = STORAGE_DIR / "chunks.jsonl"      # chunk text + metadata

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_WORDS = 180        # ~ a paragraph and a half; good for research prose
CHUNK_OVERLAP_WORDS = 40      # keeps sentences that straddle a chunk boundary searchable

# ---------------------------------------------------------------------------
# Dense retrieval (embeddings)
# ---------------------------------------------------------------------------
# Small, fast, strong general-purpose sentence embedding model (~90MB).
# Runs comfortably on CPU. No API key required — downloaded from HF Hub.
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Sparse retrieval (BM25)
# ---------------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75

# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------
# How many candidates each retriever contributes before fusion/rerank.
DENSE_TOP_K = 25
BM25_TOP_K = 25

# Reciprocal Rank Fusion constant (standard default = 60).
RRF_K = 60

# ---------------------------------------------------------------------------
# Re-ranking (cross-encoder)
# ---------------------------------------------------------------------------
# Cross-encoders score a (query, passage) pair jointly, which is far more
# accurate than comparing two independently-computed vectors. We use it to
# re-order the fused candidate list before it goes to the answer generator.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5   # final number of passages handed to the answer stage

# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
# No LLM API key is used anywhere in this project. By default the system
# runs in EXTRACTIVE mode: it composes an answer directly from the verified
# passages returned by the retriever, with no risk of hallucination. If you
# additionally want a natural-language summary, set USE_LOCAL_GENERATOR=True
# to enable a small local seq2seq model (still no API key, runs on CPU).
USE_LOCAL_GENERATOR = False
LOCAL_GENERATOR_MODEL_NAME = "google/flan-t5-base"

# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------
# Minimum cosine similarity between a generated sentence and the source
# chunk it claims to cite. Below this, the citation is flagged UNVERIFIED
# instead of silently trusted.
CITATION_SIM_THRESHOLD = 0.55
