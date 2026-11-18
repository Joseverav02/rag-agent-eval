"""CLI: compara los retrievers sobre el conjunto de evaluación, por nivel de dificultad.

    PYTHONPATH=src python -m rag --k 3 --out reports/retrieval_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.retriever import BM25Retriever, Document, TfidfRetriever, evaluate

DATA = Path(__file__).resolve().parents[2] / "data"
RETRIEVERS = (TfidfRetriever, BM25Retriever)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa retrievers RAG.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, help="JSON de salida opcional")
    args = parser.parse_args(argv)

    docs = [Document(d["id"], d["text"]) for d in load(DATA / "corpus.json")]
    queries = load(DATA / "eval_queries.json")

    subsets = {"all": queries}
    for level in sorted({q["difficulty"] for q in queries}):
        subsets[level] = [q for q in queries if q["difficulty"] == level]

    results = []
    for cls in RETRIEVERS:
        retriever = cls(docs)
        for name, subset in subsets.items():
            row = evaluate(retriever, subset, k=args.k)
            row["subset"] = name
            results.append(row)

    payload = {"k": args.k, "n_docs": len(docs), "results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
