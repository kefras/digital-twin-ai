
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the embedding model only once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_semantic_passages(pages, question, top_k=3):
    """
    Retrieve PDF pages based on semantic similarity.

    Each page must contain:
        source: PDF filename
        page: page number
        text: extracted page content
    """

    if not question or not question.strip():
        return []

    valid_pages = [
        page for page in pages
        if page.get("text", "").strip()
    ]

    if not valid_pages:
        return []

    model = load_embedding_model()

    documents = [
        page["text"] for page in valid_pages
    ]

    document_embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarity_scores = np.dot(
        document_embeddings,
        question_embedding
    )

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

    return [
        valid_pages[int(index)]
        for index in ranked_indices
        if similarity_scores[index] > 0.15
    ]
