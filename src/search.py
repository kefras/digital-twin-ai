"""Semantic nearest-neighbor search for trained Word2Vec and GloVe models."""
import math
from collections.abc import Mapping


def cosine_similarity(a, b):
    """Cosine similarity between equal-length numeric vectors."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError('Vectors must be nonempty and equal in length')
    if not all(math.isfinite(float(x)) for x in list(a) + list(b)):
        raise ValueError('Vectors must contain finite numbers')
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    if na == 0 or nb == 0:
        raise ValueError('Zero vectors have undefined cosine similarity')
    return max(-1.0, min(1.0, dot / (na * nb)))


class DigitalTwinSearch:
    """Search Word2Vec KeyedVectors or a mapping of GloVe word vectors.

    `glove_vectors` should be a dict mapping words to numeric vectors.
    For a custom GloVe class, extract vectors using its vocabulary/index.
    """

    def __init__(self, word2vec=None, glove_vectors=None):
        if word2vec is None and glove_vectors is None:
            raise ValueError('Provide at least one embedding model')
        if glove_vectors is not None and not isinstance(glove_vectors, Mapping):
            raise TypeError('glove_vectors must be a word-to-vector mapping')
        self.word2vec = word2vec
        self.glove_vectors = glove_vectors

    def search(self, query, model='word2vec', top_k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be a nonempty string')
        if not isinstance(model, str):
            raise TypeError('Model name must be a string')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        query = query.strip().lower()
        model = model.strip().lower()
        if model == 'word2vec':
            if self.word2vec is None:
                raise ValueError('Word2Vec model not loaded')
            kv = getattr(self.word2vec, 'wv', self.word2vec)
            if query not in kv:
                raise ValueError(f'Unknown word: {query}')
            return [(word, float(score)) for word, score in kv.most_similar(query, topn=top_k)]
        if model == 'glove':
            if self.glove_vectors is None:
                raise ValueError('GloVe vectors not loaded')
            if query not in self.glove_vectors:
                raise ValueError(f'Unknown word: {query}')
            scores = [(word, cosine_similarity(self.glove_vectors[query], vector))
                      for word, vector in self.glove_vectors.items() if word != query]
            return sorted(scores, key=lambda item: (-item[1], item[0]))[:top_k]
        raise ValueError("Model must be 'word2vec' or 'glove'")
