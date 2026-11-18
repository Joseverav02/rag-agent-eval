"""Agente sencillo en varias etapas: enrutar -> recuperar -> responder -> verificar.

Cada etapa es una función que recibe y devuelve un estado (dict). El generador de
respuestas es intercambiable: por defecto es extractivo (sin LLM); en producción
se sustituye por una llamada a un modelo (p. ej. Claude a través de la API).
"""

from dataclasses import dataclass, field
from typing import Callable

from rag.retriever import TfidfRetriever

Answerer = Callable[[str, list[str]], str]


@dataclass
class AgentState:
    question: str
    route: str = ""
    passages: list[str] = field(default_factory=list)
    answer: str = ""
    grounded: bool = False
    trace: list[str] = field(default_factory=list)


def extractive_answerer(question: str, passages: list[str]) -> str:
    """Baseline sin LLM: devuelve el pasaje con más solapamiento léxico con la pregunta."""
    q_terms = set(question.lower().split())
    if not passages:
        return ""
    return max(passages, key=lambda p: len(q_terms & set(p.lower().split())))


class RagAgent:
    def __init__(self, retriever: TfidfRetriever, answerer: Answerer = extractive_answerer, k: int = 2):
        self.retriever = retriever
        self.answerer = answerer
        self.k = k

    def route(self, state: AgentState) -> AgentState:
        # Regla simple: preguntas de cifras pasan por el mismo camino; aquí se distingue solo para trazar.
        state.route = "numeric" if any(ch.isdigit() for ch in state.question) else "general"
        state.trace.append(f"route={state.route}")
        return state

    def retrieve(self, state: AgentState) -> AgentState:
        hits = self.retriever.search(state.question, k=self.k)
        state.passages = [self.retriever.documents[self._index(doc_id)].text for doc_id, _ in hits]
        state.trace.append(f"retrieved={len(state.passages)}")
        return state

    def answer(self, state: AgentState) -> AgentState:
        state.answer = self.answerer(state.question, state.passages)
        state.trace.append("answered")
        return state

    def verify(self, state: AgentState) -> AgentState:
        # Verificación básica: la respuesta debe provenir de un pasaje recuperado.
        state.grounded = bool(state.answer) and state.answer in state.passages
        state.trace.append(f"grounded={state.grounded}")
        return state

    def run(self, question: str) -> AgentState:
        state = AgentState(question=question)
        for step in (self.route, self.retrieve, self.answer, self.verify):
            state = step(state)
        return state

    def _index(self, doc_id: str) -> int:
        return next(i for i, d in enumerate(self.retriever.documents) if d.doc_id == doc_id)
