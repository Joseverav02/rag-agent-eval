"""Recuperación de documentos (RAG) con TF-IDF y métricas de evaluación de recuperación.

No requiere claves de API: sirve como baseline reproducible antes de usar embeddings o un LLM.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Document:
    doc_id: str
    text: str


class TfidfRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d.text for d in documents])

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in top]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(retriever: TfidfRetriever, queries: list[dict], k: int = 3) -> dict:
    """queries: [{"question": str, "relevant": [doc_id, ...]}, ...]"""
    recalls, rr = [], []
    for item in queries:
        retrieved = [doc_id for doc_id, _ in retriever.search(item["question"], k=k)]
        relevant = set(item["relevant"])
        recalls.append(recall_at_k(retrieved, relevant, k))
        rr.append(mrr(retrieved, relevant))
    return {"recall_at_k": float(np.mean(recalls)), "mrr": float(np.mean(rr)), "n_queries": len(queries)}
