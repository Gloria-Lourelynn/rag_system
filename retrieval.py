"""
retrieval.py
------------
Implements the hybrid search + re-ranking pipeline:

    query --> dense search  ----\
                                  >-- Reciprocal Rank Fusion --> cross-encoder rerank --> top-N passages
    query --> BM25 search   ----/

Why hybrid?
- Dense (embedding) search finds semantically related passages even
  when the wording differs from the query.
- BM25 finds passages that share exact terminology (model names, greek
  symbols spelled out, acronyms) which embeddings sometimes smear over.
Combining both, then re-ranking with a cross-encoder that reads the
query and passage together, consistently beats either method alone.
"""

from dataclasses import dataclass
from typing import List

from sentence_transformers import CrossEncoder

import config
from ingest import Chunk
from indexing import HybridIndex


@dataclass
class RetrievedPassage:
    chunk: Chunk
    dense_rank: int = None       # 1-indexed rank in dense results (None if absent)
    bm25_rank: int = None        # 1-indexed rank in BM25 results (None if absent)
    fusion_score: float = 0.0    # Reciprocal Rank Fusion score
    rerank_score: float = None   # cross-encoder relevance score (set after rerank)


class HybridRetriever:
    def __init__(self, index: HybridIndex, reranker_model_name: str = config.RERANKER_MODEL_NAME):
        self.index = index
        print(f"Loading cross-encoder reranker '{reranker_model_name}' ...")
        self.reranker = CrossEncoder(reranker_model_name)

    # ------------------------------------------------------------------
    def _reciprocal_rank_fusion(self, dense_hits, bm25_hits, k: int = config.RRF_K):
        """Combine two ranked lists into one fused ranking.

        RRF score for a document = sum over each list it appears in of
        1 / (k + rank_in_that_list). It needs no score normalization
        between dense cosine similarity and BM25's unbounded scores,
        which is exactly why it's the standard way to fuse heterogeneous
        rankers.
        """
        scores = {}
        dense_rank_map = {}
        bm25_rank_map = {}

        for rank, (idx, _score) in enumerate(dense_hits, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            dense_rank_map[idx] = rank

        for rank, (idx, _score) in enumerate(bm25_hits, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            bm25_rank_map[idx] = rank

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused, dense_rank_map, bm25_rank_map

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_n: int = config.RERANK_TOP_N) -> List[RetrievedPassage]:
        dense_hits = self.index.dense_search(query, config.DENSE_TOP_K)
        bm25_hits = self.index.bm25_search(query, config.BM25_TOP_K)

        fused, dense_rank_map, bm25_rank_map = self._reciprocal_rank_fusion(dense_hits, bm25_hits)

        # Take a generous candidate pool into the reranker (cheap dense/BM25
        # stage narrows millions -> dozens; the expensive cross-encoder then
        # narrows dozens -> the final handful).
        candidate_pool = fused[:max(top_n * 4, 20)]

        passages = []
        for idx, fusion_score in candidate_pool:
            passages.append(RetrievedPassage(
                chunk=self.index.chunks[idx],
                dense_rank=dense_rank_map.get(idx),
                bm25_rank=bm25_rank_map.get(idx),
                fusion_score=fusion_score,
            ))

        if not passages:
            return []

        # Cross-encoder re-ranking: scores (query, passage) jointly.
        pairs = [(query, p.chunk.text) for p in passages]
        rerank_scores = self.reranker.predict(pairs)
        for p, s in zip(passages, rerank_scores):
            p.rerank_score = float(s)

        passages.sort(key=lambda p: p.rerank_score, reverse=True)
        return passages[:top_n]
