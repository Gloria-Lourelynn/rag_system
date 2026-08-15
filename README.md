# Local Hybrid-Search RAG for Research Papers

A retrieval-augmented question-answering system for PDF/text research papers,
built entirely from **local, open-source models**. There is no OpenAI /
Anthropic / any other paid API key anywhere in this project — every model is
pulled once, for free, from the Hugging Face Hub, cached locally, and then
runs fully offline on CPU.

```
                 ┌───────────────┐
   query ───────▶│  Dense search │──┐
                 │  (embeddings) │  │
                 └───────────────┘  │       ┌─────────────────┐      ┌──────────────────────┐
                                     ├──────▶│ Reciprocal Rank │────▶│  Cross-Encoder Rerank  │───▶ top-N passages
                 ┌───────────────┐  │       │     Fusion       │      └──────────┬───────────┘
   query ───────▶│  BM25 search  │──┘       └─────────────────┘                 │
                 │   (keyword)   │                                              ▼
                 └───────────────┘                                  ┌───────────────────────┐
                                                                      │  Answer + Citation     │
                                                                      │  Verification          │
                                                                      └───────────────────────┘
```

## Why hybrid search + reranking?

| Stage | What it's good at | What it misses |
|---|---|---|
| **Dense (embeddings)** | Semantic/paraphrase matches ("cost function" ≈ "loss") | Exact terminology, model names, symbols |
| **BM25 (keyword)** | Exact terms, acronyms, equation variables | Paraphrases, synonyms |
| **RRF fusion** | Combines both rankings without needing to normalize incompatible score scales | — |
| **Cross-encoder rerank** | Reads the query and passage *together*, so it catches subtle relevance that neither retriever alone can score well | Too slow to run over the whole corpus, so it only re-scores the fused shortlist |

This four-stage pipeline is the standard recipe used in production RAG
systems, implemented here with zero paid dependencies.

## Verified citations

The system doesn't just print `[Source: paper.pdf]` — it checks the claim:

- **Extractive mode (default):** the answer is built directly out of
  sentences copied verbatim from the retrieved chunks. Every citation is
  trivially correct because nothing is paraphrased or invented.
- **Abstractive mode (optional):** a small local model (`flan-t5-base`)
  paraphrases the retrieved passages into a fluent answer. Because a
  generative model *can* say something the source doesn't support, every
  generated sentence is re-embedded and compared (cosine similarity) against
  the chunk it claims to cite. If the similarity falls below
  `config.CITATION_SIM_THRESHOLD`, the sentence is flagged:

  ```
  The model was later used to train elephants to play chess. [1]  ⚠ UNVERIFIED
  ```

  This is the difference between a system that *displays* citations and one
  that *checks* them.

## Project layout

```
rag_system/
├── config.py        # all tunable settings (models, chunk size, thresholds)
├── ingest.py         # PDF/txt -> cleaned, overlapping chunks (with page/source metadata)
├── indexing.py       # builds & persists the FAISS dense index + BM25 sparse index
├── retrieval.py       # hybrid search: RRF fusion + cross-encoder reranking
├── generation.py      # extractive/abstractive answers + citation verification
├── pipeline.py         # RAGPipeline: wires ingest -> index -> retrieve -> generate
├── cli.py              # command-line interface (index / query / chat)
├── requirements.txt
├── data/papers/         # <- put your research paper .pdf / .txt files here
└── storage/              # <- built indexes are saved here (auto-created)
```

## Setup

```bash
pip install -r requirements.txt
```

The first time you run indexing or a query, `sentence-transformers` will
download two small models from the Hugging Face Hub (no key required):

- `all-MiniLM-L6-v2` (~90 MB) — dense embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB) — reranking

They're cached in `~/.cache/huggingface` afterward, so every run after the
first is fully offline. If you enable abstractive mode
(`config.USE_LOCAL_GENERATOR = True`), it will also download
`google/flan-t5-base` (~1 GB) the first time it's used.

## Usage

**1. Add papers.** Drop PDF or `.txt` files into `data/papers/`. Two sample
papers (plain-text excerpts of "Attention Is All You Need" and "BERT") are
included so you can try the system immediately.

**2. Build the index:**

```bash
python cli.py index
```

**3. Ask questions:**

```bash
python cli.py query "What dataset was used to pretrain BERT?"
```

Or start an interactive session:

```bash
python cli.py chat
```

**4. Or use it as a library:**

```python
from pipeline import RAGPipeline

rag = RAGPipeline()
rag.load_index()          # or rag.build_index() the first time / after adding papers

answer = rag.query("What BLEU score did the Transformer achieve on WMT 2014 En-De?")
print(answer.render())

# Programmatic access to structure + verification flags:
for s in answer.sentences:
    print(s.text, s.citation_marker, s.verified)
for c in answer.citations:
    print(c.marker, c.source, c.page)
```

## Configuration

Everything tunable lives in `config.py`:

- `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` — how papers are split
- `DENSE_TOP_K` / `BM25_TOP_K` — candidates pulled from each retriever before fusion
- `RRF_K` — Reciprocal Rank Fusion constant (60 is the standard default)
- `RERANK_TOP_N` — final number of passages used to build the answer
- `USE_LOCAL_GENERATOR` — `False` (extractive, always-grounded) or `True` (abstractive + verified)
- `CITATION_SIM_THRESHOLD` — how strict citation verification is (0–1, cosine similarity)

## Re-indexing after adding papers

Indexes aren't updated incrementally — running `python cli.py index` again
rebuilds them from everything currently in `data/papers/`. For a handful of
papers this takes seconds to a couple of minutes on CPU (dominated by
embedding time), which is why it's a separate step from querying.

## Design notes / limitations

- **Chunking is word-based with overlap**, not sentence-boundary-aware
  splitting — simple and robust across messy PDF text extraction, at the
  cost of occasionally cutting a chunk mid-sentence (the overlap window
  mitigates this for retrieval; the extractive answer stage re-splits each
  chunk into sentences before quoting, so answers themselves are still
  clean, complete sentences).
- **No external NLTK data downloads** — sentence splitting and tokenization
  use small hand-written regexes instead, so the system doesn't depend on
  extra network resources beyond the two/three HF model downloads above.
- **PDF extraction quality** depends on `pdfplumber`, which handles most
  born-digital academic PDFs well but will struggle with scanned
  (image-only) papers — those would need OCR first, which isn't included
  here.
- **Everything runs on CPU** by default and is sized (MiniLM-class models)
  to stay fast without a GPU. If you have a GPU, `sentence-transformers` and
  `transformers` will use it automatically.
