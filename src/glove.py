"""Original SimpleGloVe implementation extracted from the submitted Colab notebook."""
import numpy as np

class SimpleGloVe:
    """
    Simplified GloVe implementation using NumPy.
    Learns embeddings from word co-occurrence counts.
    """

    def __init__(self, vector_size=20, window_size=2,
                 learning_rate=0.03, epochs=300, seed=42):

        if type(vector_size) is not int or vector_size < 1:
            raise ValueError("vector_size must be positive")

        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size must be positive")

        if learning_rate <= 0 or epochs < 1:
            raise ValueError("Invalid training parameters")

        self.vector_size = vector_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

    def fit(self, sentences):
        if not isinstance(sentences, list) or not sentences:
            raise ValueError("Provide non-empty sentences")

        if any(
            not isinstance(s, list)
            or not s
            or any(not isinstance(w, str) or not w.strip()
                   for w in s)
            for s in sentences
        ):
            raise ValueError("Sentences must contain valid words")

        self.vocabulary = sorted({
            word.lower()
            for sentence in sentences
            for word in sentence
        })

        self.word_index = {
            word: i for i, word in enumerate(self.vocabulary)
        }

        size = len(self.vocabulary)
        X = np.zeros((size, size))

        # Build weighted co-occurrence matrix
        for sentence in sentences:
            words = [w.lower() for w in sentence]

            for i, word in enumerate(words):
                for j in range(
                    max(0, i - self.window_size),
                    min(len(words), i + self.window_size + 1)
                ):
                    if i != j:
                        a = self.word_index[word]
                        b = self.word_index[words[j]]
                        X[a, b] += 1.0 / abs(i - j)

        rng = np.random.default_rng(self.seed)

        W = rng.normal(0, 0.01, (size, self.vector_size))
        C = rng.normal(0, 0.01, (size, self.vector_size))
        bw = np.zeros(size)
        bc = np.zeros(size)

        rows, cols = np.nonzero(X)
        counts = X[rows, cols]

        weights = np.minimum(
            (counts / 100.0) ** 0.75, 1.0
        )
        targets = np.log(counts)

        self.loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0

            for k in rng.permutation(len(rows)):
                i, j = rows[k], cols[k]

                error = (
                    np.dot(W[i], C[j])
                    + bw[i] + bc[j]
                    - targets[k]
                )

                weight = weights[k]
                total_loss += weight * error ** 2

                gradient = 2 * weight * error

                wi = W[i].copy()
                cj = C[j].copy()

                W[i] -= self.learning_rate * gradient * cj
                C[j] -= self.learning_rate * gradient * wi

                bw[i] -= self.learning_rate * gradient
                bc[j] -= self.learning_rate * gradient

            self.loss_history.append(total_loss)

        self.embeddings = W + C

        return self

    def similarity(self, word1, word2):
        if not hasattr(self, "embeddings"):
            raise ValueError("Train the model first")

        if not isinstance(word1, str) or not isinstance(word2, str):
            raise TypeError("Words must be strings")

        word1, word2 = word1.lower(), word2.lower()

        for word in (word1, word2):
            if word not in self.word_index:
                raise ValueError(f"Unknown word: {word}")

        v1 = self.embeddings[self.word_index[word1]]
        v2 = self.embeddings[self.word_index[word2]]

        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denominator == 0:
            return 0.0

        return float(np.dot(v1, v2) / denominator)
