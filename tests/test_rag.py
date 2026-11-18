import json
from pathlib import Path

import pytest

from rag.agent import RagAgent, extractive_answerer
from rag.retriever import (
    BM25Retriever,
    Document,
    TfidfRetriever,
    evaluate,
    recall_at_k,
    reciprocal_rank,
)

DATA = Path(__file__).resolve().parents[1] / "data"

DOCS = [
    Document("d1", "La prima del seguro de auto depende de la edad y el historial del conductor."),
    Document("d2", "LoRA entrena pocas matrices adaptadoras y reduce la memoria del fine-tuning."),
    Document("d3", "Un pipeline ETL valida los datos antes de cargarlos en el almacén."),
    Document("d4", "Recall@k mide la fracción de documentos relevantes en los primeros k resultados."),
]


@pytest.fixture(params=[TfidfRetriever, BM25Retriever], ids=["tfidf", "bm25"])
def retriever(request):
    return request.param(DOCS)


def test_most_relevant_document_ranks_first(retriever):
    assert retriever.search("LoRA fine-tuning memoria", k=1)[0][0] == "d2"


def test_search_returns_k_results_sorted_by_score(retriever):
    hits = retriever.search("seguro de auto", k=3)
    assert len(hits) == 3
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        TfidfRetriever([])
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_recall_and_reciprocal_rank_values():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, k=2) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_recall_requires_relevant_docs():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_evaluate_on_easy_queries(retriever):
    queries = [
        {"id": "q1", "question": "prima seguro auto edad", "relevant": ["d1"]},
        {"id": "q2", "question": "LoRA memoria fine-tuning", "relevant": ["d2"]},
    ]
    report = evaluate(retriever, queries, k=2)
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["queries_with_miss"] == []


def test_evaluate_reports_misses():
    retriever = TfidfRetriever(DOCS)
    queries = [{"id": "q9", "question": "xyzzy", "relevant": ["d3"]}]
    report = evaluate(retriever, queries, k=1)
    assert report["recall_at_k"] == 0.0
    assert report["queries_with_miss"] == ["q9"]


def test_evaluate_rejects_empty_query_set(retriever):
    with pytest.raises(ValueError):
        evaluate(retriever, [], k=1)


def test_agent_runs_all_steps_and_is_grounded(retriever):
    agent = RagAgent(retriever, answerer=extractive_answerer, k=2)
    state = agent.run("¿Qué afecta a la prima del seguro de auto?")
    assert state.trace == ["route=general", "retrieved=2", "answered", "grounded=True"]
    assert "prima" in state.answer.lower()
    assert len(state.doc_ids) == 2


def test_agent_routes_numeric_questions():
    agent = RagAgent(TfidfRetriever(DOCS), k=1)
    assert agent.run("¿Qué es Recall@3?").route == "numeric"


def test_agent_rejects_empty_question_and_bad_k():
    agent = RagAgent(TfidfRetriever(DOCS), k=1)
    with pytest.raises(ValueError):
        agent.run("   ")
    with pytest.raises(ValueError):
        RagAgent(TfidfRetriever(DOCS), k=0)


def test_empty_passages_give_empty_answer():
    assert extractive_answerer("cualquier cosa", []) == ""


def test_bundled_eval_set_is_well_formed():
    corpus = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA / "eval_queries.json").read_text(encoding="utf-8"))
    ids = {d["id"] for d in corpus}
    assert len(ids) == len(corpus)
    for q in queries:
        assert set(q["relevant"]) <= ids
        assert q["difficulty"] in {"lexical", "paraphrase"}
