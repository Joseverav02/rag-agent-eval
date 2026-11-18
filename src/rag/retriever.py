"""Recuperación de documentos para RAG y métricas de evaluación de la recuperación.

Incluye dos baselines sin claves de API:
- ``TfidfRetriever``: TF-IDF con similitud coseno.
- ``BM25Retriever``: Okapi BM25, habitualmente más fuerte que TF-IDF en recuperación léxica.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("Se necesita al menos un documento")
        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d.text for d in self.documents])

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in top]


class BM25Retriever:
    name = "bm25"

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("Se necesita al menos un documento")
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self._tokens = [_tokenize(d.text) for d in self.documents]
        self._avgdl = sum(len(t) for t in self._tokens) / len(self._tokens)
        self._df: Counter[str] = Counter()
        for toks in self._tokens:
            self._df.update(set(toks))
        self._n = len(self.documents)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        scores = []
        for toks in self._tokens:
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                f = tf[term]
                s += self._idf(term) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
            scores.append(s)
        order = np.argsort(-np.array(scores), kind="stable")[:k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in order]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("Cada consulta necesita al menos un documento relevante")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(retriever, queries: list[dict], k: int = 3) -> dict:
    """queries: [{"id": str, "question": str, "relevant": [doc_id, ...]}, ...]"""
    if not queries:
        raise ValueError("No hay consultas para evaluar")
    recalls, rr, misses = [], [], []
    for item in queries:
        retrieved = [doc_id for doc_id, _ in retriever.search(item["question"], k=k)]
        relevant = set(item["relevant"])
        recalls.append(recall_at_k(retrieved, relevant, k))
        rr.append(reciprocal_rank(retrieved, relevant))
        if recalls[-1] < 1.0:
            misses.append(item.get("id", item["question"]))
    return {
        "retriever": retriever.name,
        "k": k,
        "n_queries": len(queries),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(rr)), 4),
        "queries_with_miss": misses,
    }
