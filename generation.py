"""
generation.py
-------------
Turns a ranked list of retrieved passages into a final answer with
CITATIONS THAT ARE ACTUALLY VERIFIED against the source text.

Two answer modes:

  EXTRACTIVE (default, config.USE_LOCAL_GENERATOR = False)
      The answer is composed directly out of the retrieved sentences.
      Every claim is, by construction, a verbatim excerpt from a real
      chunk, so citations are always 100% grounded. This mode needs no
      generative model at all.

  ABSTRACTIVE (config.USE_LOCAL_GENERATOR = True)
      A small local seq2seq model (flan-t5) paraphrases/summarizes the
      passages into a fluent answer. Because a generative model *can*
      hallucinate details not present in the source, we run a citation
      verification pass afterward: every sentence of the generated
      answer is compared (via embedding similarity) against the chunk
      it's supposed to be citing, and flagged as UNVERIFIED if it
      doesn't actually match. This is what "verified citations" means
      in this project -- we don't just print a source, we check it.

No OpenAI/Anthropic/any external API key is used in either mode.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util

import config
from retrieval import RetrievedPassage


@dataclass
class Citation:
    marker: str            # e.g. "[1]"
    source: str            # paper filename
    page: int
    chunk_id: str
    passage_excerpt: str   # short excerpt of the actual source text


@dataclass
class AnswerSentence:
    text: str
    citation_marker: Optional[str]   # which [n] this sentence claims to be backed by
    verified: Optional[bool]         # True/False/None (None = no citation claimed)
    similarity: Optional[float]      # similarity score used for verification


@dataclass
class RAGAnswer:
    query: str
    mode: str                        # "extractive" or "abstractive"
    sentences: List[AnswerSentence]
    citations: List[Citation]

    def render(self) -> str:
        """Render a human-readable answer with inline [n] markers and a
        source list, flagging anything that failed citation verification."""
        lines = []
        for s in self.sentences:
            tag = ""
            if s.citation_marker:
                flag = "" if s.verified else "  ⚠ UNVERIFIED"
                tag = f" {s.citation_marker}{flag}"
            lines.append(s.text + tag)
        body = " ".join(lines)

        src_lines = ["", "Sources:"]
        for c in self.citations:
            src_lines.append(
                f"  {c.marker} {c.source}, p.{c.page} — \"{c.passage_excerpt}\""
            )
        return body + "\n" + "\n".join(src_lines)


_SENTENCE_BOUNDARY = re.compile(r'[.!?](?:\s*\[\d+\])?\s+(?=[A-Z0-9])')


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (avoids needing an NLTK data download).

    Splits on sentence-ending punctuation followed by a capital letter or
    digit. A trailing citation marker like "[1]" right after the period is
    treated as part of the same sentence rather than a false boundary, so
    "Result X. [1] Result Y. [2]" splits into two sentences, not a mangled
    merge, and each keeps its own marker.
    """
    text = text.strip()
    if not text:
        return []
    parts = []
    last = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        boundary_text = m.group(0)
        marker_len = boundary_text.rfind(']') + 1 if ']' in boundary_text else 1
        cut = m.start() + marker_len
        parts.append(text[last:cut].strip())
        last = m.end()
    parts.append(text[last:].strip())
    return [p for p in parts if p]


class AnswerGenerator:
    def __init__(self, embedder: Optional[SentenceTransformer] = None):
        # Reuse an existing embedder if the caller has one loaded already
        # (saves loading the model twice).
        self.embedder = embedder or SentenceTransformer(config.DENSE_MODEL_NAME)
        self._gen_pipeline = None  # lazy-loaded only if abstractive mode is used

    # ------------------------------------------------------------------
    def _load_generator(self):
        if self._gen_pipeline is None:
            from transformers import pipeline
            print(f"Loading local generator '{config.LOCAL_GENERATOR_MODEL_NAME}' ...")
            self._gen_pipeline = pipeline(
                "text2text-generation",
                model=config.LOCAL_GENERATOR_MODEL_NAME,
            )
        return self._gen_pipeline

    # ------------------------------------------------------------------
    def _build_citations(self, passages: List[RetrievedPassage]) -> List[Citation]:
        citations = []
        for i, p in enumerate(passages, start=1):
            excerpt = p.chunk.text[:140].rsplit(" ", 1)[0] + "..."
            citations.append(Citation(
                marker=f"[{i}]",
                source=p.chunk.source,
                page=p.chunk.page,
                chunk_id=p.chunk.chunk_id,
                passage_excerpt=excerpt,
            ))
        return citations

    # ------------------------------------------------------------------
    def _extractive_answer(self, query: str, passages: List[RetrievedPassage],
                            citations: List[Citation]) -> List[AnswerSentence]:
        """Compose an answer out of the single most relevant sentence from
        each top passage. Since these sentences are copied verbatim from
        the source chunk, verification is always True by construction."""
        sentences = []
        for p, c in zip(passages, citations):
            candidate_sents = _split_sentences(p.chunk.text)
            if not candidate_sents:
                continue
            # Pick the sentence within the chunk most relevant to the query.
            sent_embs = self.embedder.encode(candidate_sents, normalize_embeddings=True)
            q_emb = self.embedder.encode([query], normalize_embeddings=True)
            sims = util.cos_sim(q_emb, sent_embs)[0].tolist()
            best_i = int(np.argmax(sims))
            best_sentence = candidate_sents[best_i]

            sentences.append(AnswerSentence(
                text=best_sentence,
                citation_marker=c.marker,
                verified=True,           # verbatim excerpt -> trivially grounded
                similarity=1.0,
            ))
        return sentences

    # ------------------------------------------------------------------
    def _abstractive_answer(self, query: str, passages: List[RetrievedPassage],
                             citations: List[Citation]) -> List[AnswerSentence]:
        """Generate a fluent paraphrase with a local seq2seq model, then
        verify each generated sentence against its cited source chunk."""
        gen = self._load_generator()

        context_block = "\n".join(
            f"{c.marker} {p.chunk.text}" for p, c in zip(passages, citations)
        )
        prompt = (
            "Answer the question using only the numbered context passages. "
            "After every claim, include the matching [n] citation marker.\n\n"
            f"Question: {query}\n\nContext:\n{context_block}\n\nAnswer:"
        )
        raw_output = gen(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]

        gen_sentences = _split_sentences(raw_output)
        chunk_lookup = {c.marker: p.chunk for p, c in zip(passages, citations)}

        results = []
        for sent in gen_sentences:
            marker_match = re.search(r"\[(\d+)\]", sent)
            marker = f"[{marker_match.group(1)}]" if marker_match else None
            clean_sent = re.sub(r"\s*\[\d+\]\s*", "", sent).strip()

            verified, sim = None, None
            if marker and marker in chunk_lookup:
                sent_emb = self.embedder.encode([clean_sent], normalize_embeddings=True)
                chunk_emb = self.embedder.encode([chunk_lookup[marker].text], normalize_embeddings=True)
                sim = float(util.cos_sim(sent_emb, chunk_emb)[0][0])
                verified = sim >= config.CITATION_SIM_THRESHOLD
            elif marker and marker not in chunk_lookup:
                verified = False  # cited a source that doesn't exist -> definitely flag it

            results.append(AnswerSentence(
                text=clean_sent, citation_marker=marker, verified=verified, similarity=sim
            ))
        return results

    # ------------------------------------------------------------------
    def generate(self, query: str, passages: List[RetrievedPassage]) -> RAGAnswer:
        if not passages:
            return RAGAnswer(query=query, mode="none", sentences=[
                AnswerSentence(text="No relevant passages were found in the indexed papers.",
                                citation_marker=None, verified=None, similarity=None)
            ], citations=[])

        citations = self._build_citations(passages)

        if config.USE_LOCAL_GENERATOR:
            sentences = self._abstractive_answer(query, passages, citations)
            mode = "abstractive"
        else:
            sentences = self._extractive_answer(query, passages, citations)
            mode = "extractive"

        return RAGAnswer(query=query, mode=mode, sentences=sentences, citations=citations)
