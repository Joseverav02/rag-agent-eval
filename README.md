# rag-agent-eval

[![CI](https://github.com/Joseverav02/rag-agent-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Joseverav02/rag-agent-eval/actions/workflows/ci.yml)

Sistema **RAG** con un **agente en varias etapas** (enrutar → recuperar → responder → verificar) y un **arnés de evaluación** que mide la calidad de la recuperación **antes** de añadir un LLM.

Principio: no se puede mejorar lo que no se mide. Primero se evalúa el retriever; después, el generador.

## Resultados (corpus de 20 documentos, 30 consultas, k = 3)

| Retriever | Subconjunto | Recall@3 | MRR |
|---|---|---|---|
| TF-IDF | lexical (22) | 1,000 | 0,970 |
| BM25 | lexical (22) | 1,000 | 0,970 |
| TF-IDF | paráfrasis (8) | 0,625 | 0,479 |
| BM25 | paráfrasis (8) | 0,625 | 0,625 |

Los métodos léxicos fallan cuando la pregunta no comparte vocabulario con el documento. El detalle de los fallos y sus causas está en [docs/error_analysis.md](docs/error_analysis.md). Esa limitación es la razón para el siguiente paso del proyecto (embeddings densos o búsqueda híbrida).

> Datos de evaluación **escritos a mano** y de tamaño reducido. Los resultados son orientativos, no una medida de rendimiento general.

## Qué incluye
- **Retrievers sin claves de API:** TF-IDF (coseno) y BM25 (Okapi) implementados desde cero.
- **Métricas de recuperación:** Recall@k y MRR, con informe de consultas fallidas.
- **Agente con etapas explícitas** y traza de ejecución.
- **Verificación de *grounding*:** la respuesta debe proceder de un pasaje recuperado.
- **Generador intercambiable:** extractivo por defecto; se sustituye por un LLM en producción.
- **Tests:** 18 casos con `pytest`, en ambos retrievers. CI en GitHub Actions.

## Uso

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src python -m rag --k 3 --out reports/retrieval_eval.json
```

Desde código:

```python
from rag.retriever import Document, BM25Retriever
from rag.agent import RagAgent

docs = [Document("d1", "..."), Document("d2", "...")]
agent = RagAgent(BM25Retriever(docs), k=2)
state = agent.run("¿tu pregunta?")
print(state.answer, state.grounded, state.trace)
```

## Estructura

```
src/rag/
  retriever.py   # TF-IDF, BM25, Recall@k, MRR, evaluate()
  agent.py       # RagAgent: route -> retrieve -> answer -> verify
  __main__.py    # CLI de evaluación
data/
  corpus.json
  eval_queries.json
reports/retrieval_eval.json
docs/error_analysis.md
tests/test_rag.py
```

## Próximas mejoras
- Embeddings densos y búsqueda híbrida, repitiendo esta evaluación para medir la mejora.
- Generador con LLM y evaluación de *faithfulness* con un juez automático.
- Ampliar el conjunto de evaluación con consultas de terceros para reducir el sesgo del autor.
- Grafo de estados con LangGraph en lugar de la secuencia fija actual.

## Licencia
MIT
