
"""Semantic retrieval for the Digital Twin RAG system."""

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def index_documents(self, chunks):
        if not isinstance(chunks, list):
            raise TypeError("Chunks must be provided as a list.")

        if not chunks:
            raise ValueError("Chunks list cannot be empty.")

        if any(
            not isinstance(chunk, str) or not chunk.strip()
            for chunk in chunks
        ):
            raise ValueError("Each chunk must be a non-empty string.")

        self.chunks = chunks.copy()
        self.embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return len(self.chunks)

    def search(self, question, top_k=2, threshold=0.30):
        if not isinstance(question, str):
            raise TypeError("Question must be a string.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float)
        ) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be between -1 and 1.")

        if self.embeddings is None:
            raise RuntimeError("Index documents before searching.")

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        similarities = np.dot(self.embeddings, query_embedding)
        ranked_indices = np.argsort(similarities)[::-1]

        results = []

        for index in ranked_indices[:top_k]:
            score = float(similarities[index])

            if score >= threshold:
                results.append({
                    "chunk_id": int(index) + 1,
                    "text": self.chunks[index],
                    "similarity": round(score, 4)
                })

        return results
