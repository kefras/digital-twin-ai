import unittest
from src.search import DigitalTwinSearch, cosine_similarity


class FakeVectors:
    def __contains__(self, word):
        return word in {'cloud', 'aws', 'server'}

    def most_similar(self, word, topn):
        return [('aws', 0.8), ('server', 0.6)][:topn]


class TestSemanticSearch(unittest.TestCase):
    def setUp(self):
        self.search = DigitalTwinSearch(
            word2vec=FakeVectors(),
            glove_vectors={'cloud': [1, 0], 'aws': [0.9, 0.1], 'server': [0, 1]},
        )

    def test_word2vec(self):
        self.assertEqual(self.search.search(' CLOUD ', top_k=1), [('aws', 0.8)])

    def test_glove(self):
        self.assertEqual(self.search.search('cloud', model='glove')[0][0], 'aws')

    def test_unknown_word(self):
        with self.assertRaisesRegex(ValueError, 'Unknown word'):
            self.search.search('unknown')

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            self.search.search(' ')

    def test_invalid_top_k(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.search.search('cloud', top_k=value)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            self.search.search('cloud', model='invalid')

    def test_cosine(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_zero_vector(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])

    def test_no_models(self):
        with self.assertRaises(ValueError):
            DigitalTwinSearch()


if __name__ == '__main__':
    unittest.main()
