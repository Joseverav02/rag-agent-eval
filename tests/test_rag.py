import pytest

from rag.agent import RagAgent, extractive_answerer
from rag.retriever import Document, TfidfRetriever, mrr, recall_at_k, evaluate

DOCS = [
    Document("d1", "La prima anual del seguro de auto depende de la edad y el historial del conductor."),
    Document("d2", "El fine-tuning con LoRA entrena pocas matrices adaptadoras y reduce la memoria."),
    Document("d3", "Un pipeline ETL valida los datos antes de cargarlos en el almacén."),
    Document("d4", "Recall@k mide la fracción de documentos relevantes recuperados en los primeros k resultados."),
]
QUERIES = [
    {"question": "¿Qué afecta a la prima del seguro de auto?", "relevant": ["d1"]},
    {"question": "¿Cómo reduce LoRA la memoria del fine-tuning?", "relevant": ["d2"]},
    {"question": "¿Qué mide recall@k?", "relevant": ["d4"]},
]


def test_retriever_returns_most_relevant_first():
    r = TfidfRetriever(DOCS)
    top = r.search("LoRA fine-tuning memoria", k=1)
    assert top[0][0] == "d2"


def test_metrics_basic_values():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, k=2) == 0.5
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["a"], {"z"}) == 0.0


def test_evaluate_reports_high_scores_on_easy_queries():
    report = evaluate(TfidfRetriever(DOCS), QUERIES, k=2)
    assert report["recall_at_k"] >= 0.9
    assert report["mrr"] >= 0.9


def test_agent_runs_all_steps_and_is_grounded():
    agent = RagAgent(TfidfRetriever(DOCS), answerer=extractive_answerer, k=2)
    state = agent.run("¿Qué afecta a la prima del seguro de auto?")
    assert state.trace == ["route=general", "retrieved=2", "answered", "grounded=True"]
    assert "prima" in state.answer.lower()


def test_empty_passages_give_empty_answer():
    assert extractive_answerer("cualquier cosa", []) == ""
