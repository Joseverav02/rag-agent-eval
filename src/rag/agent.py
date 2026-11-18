"""Agente en varias etapas: enrutar -> recuperar -> responder -> verificar.

Cada etapa recibe y devuelve un estado. El generador de respuestas es intercambiable:
por defecto es extractivo (sin LLM); en producción se sustituye por una llamada a un modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rag.retriever import Document

Answerer = Callable[[str, list[str]], str]


@dataclass
class AgentState:
    question: str
    route: str = ""
    doc_ids: list[str] = field(default_factory=list)
    passages: list[str] = field(default_factory=list)
    answer: str = ""
    grounded: bool = False
    trace: list[str] = field(default_factory=list)


def extractive_answerer(question: str, passages: list[str]) -> str:
    """Baseline sin LLM: devuelve el pasaje con más solapamiento léxico con la pregunta."""
    if not passages:
        return ""
    q_terms = set(question.lower().split())
    return max(passages, key=lambda p: len(q_terms & set(p.lower().split())))


class RagAgent:
    def __init__(self, retriever, answerer: Answerer = extractive_answerer, k: int = 2):
        if k < 1:
            raise ValueError("k debe ser al menos 1")
        self.retriever = retriever
        self.answerer = answerer
        self.k = k
        self._by_id: dict[str, Document] = {d.doc_id: d for d in retriever.documents}

    def route(self, state: AgentState) -> AgentState:
        # Regla simple: distingue preguntas cuantitativas (con cifras) para trazar y, en el futuro, enrutar.
        state.route = "numeric" if any(ch.isdigit() for ch in state.question) else "general"
        state.trace.append(f"route={state.route}")
        return state

    def retrieve(self, state: AgentState) -> AgentState:
        hits = self.retriever.search(state.question, k=self.k)
        state.doc_ids = [doc_id for doc_id, _ in hits]
        state.passages = [self._by_id[doc_id].text for doc_id in state.doc_ids]
        state.trace.append(f"retrieved={len(state.passages)}")
        return state

    def answer(self, state: AgentState) -> AgentState:
        state.answer = self.answerer(state.question, state.passages)
        state.trace.append("answered")
        return state

    def verify(self, state: AgentState) -> AgentState:
        # Verificación de grounding: la respuesta debe coincidir con un pasaje recuperado.
        state.grounded = bool(state.answer) and state.answer in state.passages
        state.trace.append(f"grounded={state.grounded}")
        return state

    def run(self, question: str) -> AgentState:
        if not question.strip():
            raise ValueError("La pregunta no puede estar vacía")
        state = AgentState(question=question)
        for step in (self.route, self.retrieve, self.answer, self.verify):
            state = step(state)
        return state
