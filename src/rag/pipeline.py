
"""Digital Twin RAG pipeline."""

import time

from src.rag.chunker import DocumentChunker
from src.rag.retriever import SemanticRetriever


class DigitalTwinRAG:
    def __init__(self, document):
        self.chunker = DocumentChunker(
            chunk_size=8,
            overlap=2
        )

        self.retriever = SemanticRetriever()

        chunks = self.chunker.split_document(document)

        if not chunks:
            raise ValueError("Document contains no usable text.")

        self.retriever.index_documents(chunks)

        print("Digital Twin RAG initialized successfully!")

    def ask(self, question):
        if not isinstance(question, str):
            raise TypeError("Question must be a string.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        start = time.perf_counter()

        results = self.retriever.search(
            question,
            top_k=3,
            threshold=0.30
        )

        if not results:
            answer = "I don't have enough information to answer."
            status = "abstained"
        else:
            context = " ".join(
                result["text"] for result in results
            )

            answer = self.generate_answer(question, context)
            status = (
                "abstained"
                if answer == "I don't have enough information to answer."
                else "answered"
            )

        duration = time.perf_counter() - start

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": results,
            "duration_seconds": round(duration, 4),
            "status": status
        }

    @staticmethod
    def generate_answer(question, context):
        """Simple rule-based answer extraction for the demo."""

        question = question.lower()
        context = context.lower()

        rules = [
            ("terraform", ["automates", "infrastructure"]),
            ("docker", ["docker"]),
            ("kubernetes", ["kubernetes", "manage"]),
            ("aws", ["cloud computing", "platform"])
        ]

        for keyword, triggers in rules:
            if (
                keyword in context
                and any(trigger in question for trigger in triggers)
            ):
                answers = {
                    "terraform": "Terraform automates cloud infrastructure deployment.",
                    "docker": "Docker helps developers build and deploy containers.",
                    "kubernetes": "Kubernetes manages containerized applications.",
                    "aws": "AWS provides cloud computing services."
                }

                return answers[keyword]

        return "I don't have enough information to answer."
