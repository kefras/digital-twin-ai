"""Train both original notebook algorithms and search their vocabulary."""
import re
from pathlib import Path
from gensim.models import Word2Vec
from .glove import SimpleGloVe
from .search import DigitalTwinSearch

def run():
    corpus = Path(__file__).resolve().parents[1] / "data" / "original_colab_corpus.txt"
    sentences = [re.findall(r"[a-z]+", line.lower()) for line in corpus.read_text().splitlines()]
    sentences = [s for s in sentences if s]
    word2vec = Word2Vec(sentences=sentences, vector_size=50, window=3, min_count=1, workers=1, sg=1, epochs=200, seed=42)
    glove = SimpleGloVe().fit(sentences)
    vectors = {word: glove.embeddings[glove.word_index[word]] for word in glove.vocabulary}
    search = DigitalTwinSearch(word2vec=word2vec, glove_vectors=vectors)
    for model in ("word2vec", "glove"):
        print(model, search.search("cloud", model=model, top_k=5))
    return word2vec, glove

if __name__ == "__main__":
    run()
