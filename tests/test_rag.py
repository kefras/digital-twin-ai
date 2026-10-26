
import unittest
from unittest.mock import patch

from src.rag.pipeline import DigitalTwinRAG


DOCUMENT = """
AWS provides cloud computing services.
Docker helps developers build and deploy containers.
Kubernetes manages containerized applications.
Terraform automates cloud infrastructure deployment.
"""


class TestDigitalTwinRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rag = DigitalTwinRAG(DOCUMENT)

    def test_valid_question(self):
        result = self.rag.ask("What is Docker used for?")

        self.assertEqual(result["status"], "answered")
        self.assertIn("Docker", result["answer"])

    def test_terraform_question(self):
        result = self.rag.ask(
            "Which tool automates cloud infrastructure?"
        )

        self.assertEqual(result["status"], "answered")
        self.assertIn("Terraform", result["answer"])

    def test_unknown_question(self):
        result = self.rag.ask(
            "Who is the president of Mars?"
        )

        self.assertEqual(result["status"], "abstained")

    def test_empty_question(self):
        with self.assertRaises(ValueError):
            self.rag.ask("")

    def test_whitespace_question(self):
        with self.assertRaises(ValueError):
            self.rag.ask("   ")

    def test_invalid_input(self):
        for value in [None, 123, [], {}]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.rag.ask(value)

    def test_response_structure(self):
        result = self.rag.ask(
            "What does Kubernetes manage?"
        )

        expected_keys = [
            "question",
            "answer",
            "retrieved_chunks",
            "duration_seconds",
            "status",
        ]

        for key in expected_keys:
            self.assertIn(key, result)

        self.assertGreaterEqual(
            result["duration_seconds"], 0
        )

    def test_no_retrieval_results(self):
        with patch.object(
            self.rag.retriever,
            "search",
            return_value=[]
        ):
            result = self.rag.ask(
                "What is Docker used for?"
            )

        self.assertEqual(result["status"], "abstained")


if __name__ == "__main__":
    unittest.main()
