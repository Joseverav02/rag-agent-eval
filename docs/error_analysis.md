# Análisis de errores de recuperación

Configuración: corpus de 20 documentos, 30 consultas, k = 3 (`reports/retrieval_eval.json`).

## Resultados por nivel de dificultad

| Retriever | Subconjunto | Consultas | Recall@3 | MRR |
|---|---|---|---|---|
| TF-IDF | lexical | 22 | 1,000 | 0,970 |
| TF-IDF | paraphrase | 8 | 0,625 | 0,479 |
| BM25 | lexical | 22 | 1,000 | 0,970 |
| BM25 | paraphrase | 8 | 0,625 | 0,625 |

- **Lexical:** la pregunta comparte vocabulario con el documento correcto.
- **Paraphrase:** la pregunta describe el mismo concepto con palabras distintas.

## Consultas fallidas (ambos retrievers)

| Id | Pregunta | Documento relevante | Causa probable |
|---|---|---|---|
| q25 | Riesgo de que alguien manipule al asistente escondiendo órdenes en los documentos | d18 (inyección de prompts) | Sin solapamiento léxico con "inyección de prompts". |
| q28 | Cuánto debe cobrar una compañía para cubrir lo que pagará en daños | d10, d09 | Concepto de "loss ratio" descrito sin usar el término. |
| q30 | Cómo saber si el buscador pone la fuente correcta entre sus primeras sugerencias | d03 (Recall@k) | El documento usa "resultados" y "fracción"; la pregunta usa "sugerencias" y "fuente". |

## Conclusiones
1. Ambos métodos léxicos son perfectos cuando hay solapamiento de vocabulario y fallan igual en parafrasis.
2. BM25 mejora la MRR en consultas parafraseadas (0,625 frente a 0,479), pero no el recall: con solo 8 consultas esa diferencia es **orientativa**, no estadísticamente concluyente.
3. Siguiente paso lógico: embeddings densos o búsqueda híbrida (BM25 + embeddings) y repetir esta misma evaluación para medir la mejora.

## Limitaciones de la evaluación
- Corpus pequeño (20 documentos): los resultados no son extrapolables a un corpus real.
- Solo 8 consultas parafraseadas: alta varianza.
- Las consultas las ha escrito el autor, así que pueden tener sesgo hacia el corpus.
