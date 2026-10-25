"""Train a small Word2Vec demo from local example sentences."""
import argparse
from pathlib import Path
from gensim.models import Word2Vec


def train_word2vec(corpus_path, output_path):
    lines = Path(corpus_path).read_text(encoding='utf-8').splitlines()
    sentences = [line.lower().split() for line in lines if line.strip()]
    if not sentences:
        raise ValueError('Corpus must contain at least one nonempty sentence')
    model = Word2Vec(sentences=sentences, vector_size=50, window=3,
                     min_count=1, workers=1, seed=42, epochs=200)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='data/sample_corpus.txt')
    parser.add_argument('--output', default='models/word2vec.model')
    args = parser.parse_args()
    model = train_word2vec(args.corpus, args.output)
    print(f'Saved {len(model.wv)} words to {args.output}')
