# rag-agent-eval

Sistema **RAG** con un **agente en varias etapas** (enrutar → recuperar → responder → verificar) y un **arnés de evaluación** que mide la calidad de la recuperación antes de añadir un LLM.

Parte del principio de que no se puede mejorar lo que no se mide: primero se evalúa el retriever, después el generador.

## Qué demuestra
- Recuperación con TF-IDF (baseline reproducible, sin claves de API).
- Métricas de recuperación: **Recall@k** y **MRR**.
- Agente con etapas explícitas y **trazas** de ejecución.
- Verificación de *grounding*: la respuesta debe proceder de un pasaje recuperado.
- Generador intercambiable: extractivo por defecto; se puede sustituir por un LLM.
- Tests con `pytest`.

## Uso

```bash
pip install -r requirements.txt
pytest -q
```

Para usarlo desde código:

```python
from rag.retriever import Document, TfidfRetriever, evaluate
from rag.agent import RagAgent

docs = [Document("d1", "..."), Document("d2", "...")]
agent = RagAgent(TfidfRetriever(docs), k=2)
print(agent.run("¿tu pregunta?").answer)
```

## Próximas mejoras
- Embeddings densos y búsqueda híbrida (BM25 + embeddings).
- Generador con LLM y evaluación de *faithfulness* con un juez automático.
- Orquestación con LangGraph para el grafo de estados.

## Licencia
MIT
