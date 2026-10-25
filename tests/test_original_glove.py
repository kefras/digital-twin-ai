"""Integration tests for the original notebook's custom GloVe algorithm."""
import unittest
import numpy as np
from src.glove import SimpleGloVe
from src.search import DigitalTwinSearch

class TestOriginalGloVe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SimpleGloVe(vector_size=8, epochs=5, seed=42).fit([["cloud", "server", "cloud"], ["server", "network"]])

    def test_dimensions_and_finite_values(self):
        self.assertEqual(self.model.embeddings.shape, (3, 8))
        self.assertTrue(np.isfinite(self.model.embeddings).all())
        self.assertEqual(len(self.model.loss_history), 5)

    def test_unknown_word(self):
        with self.assertRaises(ValueError):
            self.model.similarity("cloud", "unknown")

    def test_search_adapter(self):
        vectors = {w: self.model.embeddings[self.model.word_index[w]] for w in self.model.vocabulary}
        search = DigitalTwinSearch(glove_vectors=vectors)
        self.assertEqual(len(search.search("cloud", model="glove", top_k=2)), 2)

    def test_invalid_training_data(self):
        with self.assertRaises(ValueError):
            SimpleGloVe().fit([])
