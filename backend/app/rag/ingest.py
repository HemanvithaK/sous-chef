import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "substitutions.json"
QDRANT_PATH = Path(__file__).parent.parent.parent / "data" / "qdrant"
COLLECTION = "substitutions"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

DOC_VECTOR = "doc"
NAME_VECTOR = "name"


def build_document(entry: dict) -> str:
    return (
        f"Ingredient: {entry['ingredient']}. "
        f"Role in cooking: {entry['role']}. "
        f"Substitutes: {', '.join(entry['substitutes'])}. "
        f"Ratio: {entry['ratio']}. "
        f"Why this works: {entry['reasoning']} "
        f"Caveats: {entry['caveats']}"
    )


def build_name_text(entry: dict) -> str:
    return f"{entry['ingredient']}, {entry['role']}"


def ingest():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} substitution entries")

    embedder = TextEmbedding(model_name=MODEL_NAME)

    doc_vectors = list(embedder.embed([build_document(e) for e in entries]))
    name_vectors = list(embedder.embed([build_name_text(e) for e in entries]))
    dim = len(doc_vectors[0])
    print(f"Embedded {len(entries)} entries x 2 views, vector dim = {dim}")

    client = QdrantClient(path=str(QDRANT_PATH))

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DOC_VECTOR: VectorParams(size=dim, distance=Distance.COSINE),
            NAME_VECTOR: VectorParams(size=dim, distance=Distance.COSINE),
        },
    )

    points = []
    for i, entry in enumerate(entries):
        points.append(
            PointStruct(
                id=i,
                vector={
                    DOC_VECTOR: doc_vectors[i].tolist(),
                    NAME_VECTOR: name_vectors[i].tolist(),
                },
                payload={
                    "ingredient": entry["ingredient"],
                    "role": entry["role"],
                    "substitutes": entry["substitutes"],
                    "ratio": entry["ratio"],
                    "reasoning": entry["reasoning"],
                    "caveats": entry["caveats"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
    client.close()

    print(f"Indexed {len(points)} points with 2 named vectors into '{COLLECTION}'")
    print(f"Vector store saved to {QDRANT_PATH}")


if __name__ == "__main__":
    ingest()