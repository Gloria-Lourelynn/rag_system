"""
pipeline.py
-----------
The single entry point that wires everything together:

    ingest -> index -> hybrid retrieve -> rerank -> generate + verify

Usage (see cli.py for the command-line wrapper):

    from pipeline import RAGPipeline
    rag = RAGPipeline()
    rag.build_index()                 # run once (or whenever papers change)
    answer = rag.query("What is the attention mechanism?")
    print(answer.render())
"""

from pathlib import Path

import config
from indexing import HybridIndex, build_and_save_index
from retrieval import HybridRetriever
from generation import AnswerGenerator, RAGAnswer


class RAGPipeline:
    def __init__(self):
        self.index: HybridIndex = None
        self.retriever: HybridRetriever = None
        self.generator: AnswerGenerator = None

    # ------------------------------------------------------------------
    def build_index(self, papers_dir: Path = config.PAPERS_DIR) -> None:
        """Ingest papers and build both search indexes from scratch."""
        self.index = build_and_save_index(papers_dir)
        self._init_retriever_and_generator()

    def load_index(self) -> None:
        """Load a previously-built index from disk (fast, no re-embedding)."""
        self.index = HybridIndex.load()
        self._init_retriever_and_generator()

    def _init_retriever_and_generator(self) -> None:
        self.retriever = HybridRetriever(self.index)
        # Reuse the same embedding model instance for citation verification
        # instead of loading a second copy into memory.
        self.generator = AnswerGenerator(embedder=self.index.embedder)

    # ------------------------------------------------------------------
    def query(self, question: str, top_n: int = config.RERANK_TOP_N) -> RAGAnswer:
        if self.retriever is None or self.generator is None:
            raise RuntimeError("Call build_index() or load_index() before query().")
        passages = self.retriever.retrieve(question, top_n=top_n)
        return self.generator.generate(question, passages)
