"""Run semantic search against a locally trained Word2Vec model."""
import argparse
from gensim.models import Word2Vec
from src.search import DigitalTwinSearch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/word2vec.model')
    parser.add_argument('--query', default='cloud')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    model = Word2Vec.load(args.model)  # Load only files from trusted sources.
    for word, score in DigitalTwinSearch(word2vec=model).search(args.query, top_k=args.top_k):
        print(f'{word:20} {score:.4f}')


if __name__ == '__main__':
    main()
