from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient


QDRANT_PATH = Path(__file__).parent.parent.parent / "data" / "qdrant"
COLLECTION = "substitutions"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

DOC_VECTOR = "doc"
NAME_VECTOR = "name"

DOC_THRESHOLD = 0.55
NAME_THRESHOLD = 0.55


class SubstitutionRetriever:
    def __init__(self):
        self._client = None
        self._embedder = None

    def _ensure_loaded(self):
        if self._client is not None:
            return
        self._embedder = TextEmbedding(model_name=MODEL_NAME)
        self._client = QdrantClient(path=str(QDRANT_PATH))

    def _embed(self, text: str) -> list[float]:
        return list(self._embedder.embed([text]))[0].tolist()

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_threshold: float = DOC_THRESHOLD,
        name_threshold: float = NAME_THRESHOLD,
        debug: bool = False,
    ) -> list[dict]:
        self._ensure_loaded()
        vector = self._embed(query)

        doc_response = self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            using=DOC_VECTOR,
            limit=top_k,
            with_payload=True,
        )

        name_response = self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            using=NAME_VECTOR,
            limit=len(SUPPORTED_IDS) if False else 50,
            with_payload=False,
        )
        name_scores = {p.id: p.score for p in name_response.points}

        hits = []
        for point in doc_response.points:
            doc_score = point.score
            name_score = name_scores.get(point.id, 0.0)

            if debug:
                print(
                    f"    {point.payload['ingredient']:<20} "
                    f"doc={doc_score:.3f} name={name_score:.3f}"
                )

            if doc_score < doc_threshold:
                continue
            if name_score < name_threshold:
                continue

            payload = point.payload
            hits.append({
                "ingredient": payload["ingredient"],
                "substitutes": payload["substitutes"],
                "ratio": payload["ratio"],
                "role": payload["role"],
                "reasoning": payload["reasoning"],
                "caveats": payload["caveats"],
                "doc_score": round(doc_score, 3),
                "name_score": round(name_score, 3),
            })

        return hits

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None


SUPPORTED_IDS = []

_retriever = SubstitutionRetriever()


def search_substitutions(query: str, top_k: int = 3, debug: bool = False) -> list[dict]:
    return _retriever.search(query, top_k=top_k, debug=debug)


def close_retriever():
    _retriever.close()


def search_substitutions_verified(
    query: str,
    top_k: int = 3,
    debug: bool = False,
) -> dict:
    from app.rag.verifier import verify_candidates

    candidates = _retriever.search(query, top_k=top_k, debug=debug)

    if not candidates:
        return {"verified": [], "candidates": [], "stage": "retrieval_empty"}

    matched_indices = verify_candidates(query, candidates)
    verified = [candidates[i] for i in matched_indices]

    return {
        "verified": verified,
        "candidates": candidates,
        "stage": "verified" if verified else "rejected_by_verifier",
    }
