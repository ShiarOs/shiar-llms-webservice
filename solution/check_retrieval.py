"""Check labeled retrieval cases with local embeddings, without generation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from solution.llm_app import rag


def main() -> None:
    cases = json.loads(Path(__file__).with_name("retrieval_cases.json").read_text())
    failures = 0
    original_path = rag.QDRANT_PERSIST_DIR
    with TemporaryDirectory(prefix="mle-retrieval-") as directory:
        rag.QDRANT_PERSIST_DIR = Path(directory) / "index"
        try:
            store = rag.load_vector_store()
            for case in cases:
                hits = store.similarity_search_with_score(case["question"], k=5)
                pages = {
                    doc.metadata["page"] + 1
                    for doc, score in hits
                    if score >= rag.MIN_RELEVANCE
                }
                expected = set(case["expected_pages"])
                passed = bool(pages & expected) if expected else not pages
                failures += not passed
                print(
                    f"{'PASS' if passed else 'FAIL'} {case['question']} "
                    f"pages={sorted(pages)} scores={[round(s, 4) for _, s in hits]}"
                )
        finally:
            rag.close_vector_store()
            rag.QDRANT_PERSIST_DIR = original_path
    if failures:
        raise SystemExit(f"{failures}/{len(cases)} retrieval checks failed")


if __name__ == "__main__":
    main()
