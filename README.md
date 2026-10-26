# Digital Twin Word Embeddings — Word2Vec & GloVe

An educational semantic-search prototype exploring Word2Vec, a custom GloVe implementation, cosine similarity, and PCA visualization, developed for a Week 2 AI School embeddings challenge.

## Project status

- **Original completed Colab notebook included:** `notebooks/Week2_08_embeddings_deep_dive_completed.ipynb`, including its saved outputs.
- **Original implementation included:** `src/glove.py` contains the `SimpleGloVe` class extracted from Stage 3 of the submitted notebook, without algorithmic changes.
- **Original sample corpus included:** `data/original_colab_corpus.txt` reproduces the ten source documents from Stage 1.
- **Runnable integration:** `src/integrated_demo.py` trains both algorithms on the original corpus and searches the vocabulary.
- **Not included:** trained `.model` and `.pkl` binaries (not part of the uploaded notebook); copy them from Colab if needed.
- **Scope:** educational word-level semantic search prototype, not a deployed Digital Twin service.

## Architecture

See [architecture diagram](diagrams/architecture.md).

## Quick start

Python 3.11 recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
python -m src.demo --query cloud --top-k 5
```

Results will vary with corpus size, training and library versions. This toy corpus is not sufficient to establish robust semantic accuracy.

## Run the original notebook implementation locally

```bash
python -m src.integrated_demo
```

The full notebook is included in `notebooks/`. It uses Colab-specific commands for installation and downloading; run those cells in Google Colab rather than a standard Python interpreter. The original GloVe class is also available as a standalone import from `src.glove`.

To reuse saved Colab model binaries, place them in `models/` (ignored by Git). Never unpickle files from untrusted sources. The notebook's `SimpleGloVe` class was defined in `__main__` when pickled, so a separate Python process may need to expose the class under that name before loading. Re-training with `src.integrated_demo` avoids this portability issue.

## Evidence from Colab screenshots

- 10 documents and 42 tokens were reported in the original preprocessing exercise.
- Word2Vec reported 29 vocabulary items and 50-dimensional vectors.
- Custom GloVe reported 29 vocabulary items and 20-dimensional vectors, with loss decreasing from approximately 0.6717 to 0.5162 over 300 epochs.
- Semantic search returned five neighbors of `cloud` for each model and handled unknown-word input.
- Saved files reported: `word2vec.model` and `glove.pkl`.

These are observed educational experiment results, not benchmarked production metrics.

## Testing

`python -m unittest discover -s tests -v` checks search ranking, case normalization, error handling, cosine similarity, and missing-model behavior. These are isolated unit tests; additional tests exercise the original custom GloVe training and search adapter.

## Future improvements

- Larger domain-specific corpus and held-out evaluation
- Phrase/sentence search and out-of-vocabulary handling
- Interactive user interface and API
- Model versioning and reproducible benchmarks
- Actual Digital Twin integration


## Digital Twin AI — Retrieval-Augmented Generation (RAG)

This project now includes a Retrieval-Augmented Generation (RAG) application that uses semantic search to retrieve relevant information from a customizable knowledge base.

### Features

- **Document chunking:** Splits knowledge-base content into smaller, overlapping sections.
- **Semantic retrieval:** Uses sentence embeddings to identify relevant text.
- **Grounded question answering:** Returns answers based on retrieved knowledge-base content.
- **Abstention handling:** Responds when insufficient relevant information is available.
- **Interactive interface:** Provides a Streamlit application for asking questions and editing knowledge-base content.
- **Automated testing:** Includes unit tests for retrieval, input validation, and question-answering behavior.

### Architecture

```text
User Question
     |
     v
Streamlit Interface
     |
     v
DigitalTwinRAG Pipeline
     |
     +--> Knowledge Base
     |        |
     |        v
     |    Document Chunker
     |        |
     |        v
     |    Sentence Embeddings
     |        |
     |        v
     |    Semantic Retriever
     |        |
     +--------+
     |
     v
Relevant Context
     |
     v
Grounded Answer / Abstention
     |
     v
Streamlit Results
```

### Installation

Use Python 3.12 and install the required packages:

```bash
python -m pip install -r requirements.txt
```

### Run the RAG Application

From the project root:

```bash
python -m streamlit run src/rag/app.py
```

Open the local URL displayed in your terminal, typically:

`http://localhost:8501`

### Run Automated Tests

```bash
python -m unittest discover -s tests -v
```

**Latest local test result:** 21 tests passed.

### Project Structure

```text
src/
  rag/
    __init__.py
    app.py
    chunker.py
    pipeline.py
    retriever.py

tests/
  test_rag.py
```

### Example Questions

- What is Docker used for?
- What does Kubernetes manage?
- Which tool automates cloud infrastructure?
- Which platform provides cloud computing services?

The application is designed to answer questions supported by the provided knowledge base and abstain when sufficient information is unavailable.

### Limitations

This implementation is a prototype. It uses semantic retrieval and grounded answer extraction rather than a full generative language model. Retrieval accuracy depends on the quality and coverage of the knowledge base.
## License

MIT — see LICENSE.
