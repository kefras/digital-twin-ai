
"""Document chunking utilities for the Digital Twin RAG system."""


class DocumentChunker:
    def __init__(self, chunk_size=100, overlap=20):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
            raise ValueError(
                "overlap must be a non-negative integer smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_document(self, document):
        if not isinstance(document, str):
            raise TypeError("Document must be a string.")

        if not document.strip():
            return []

        words = document.split()
        step = self.chunk_size - self.overlap
        chunks = []

        for start in range(0, len(words), step):
            chunk_words = words[start:start + self.chunk_size]

            if not chunk_words:
                break

            chunks.append(" ".join(chunk_words))

            if start + self.chunk_size >= len(words):
                break

        return chunks
