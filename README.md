# Digital Twin AI — Word2Vec, GloVe, RAG & Gemini PDF Chat

**An interactive AI-powered document question-answering and semantic search application built with Python, Word2Vec, GloVe, Sentence Transformers, Google Gemini, and Streamlit.**

[**Live Demo — Try Digital Twin AI**](https://digitaltwin1.streamlit.app) | [**GitHub Repository**](https://github.com/kefras/digital-twin-word2vec-glove-embeddings)

## Overview

Digital Twin AI began as an educational word embeddings project for a Week 2 AI School challenge, exploring Word2Vec, a custom GloVe implementation, cosine similarity, and PCA visualization.

The project has since evolved into an interactive AI application combining traditional word embeddings, semantic retrieval, Retrieval-Augmented Generation (RAG), and conversational PDF question answering powered by Google Gemini.

Users can explore a customizable knowledge base or upload PDF documents and ask questions based on their contents.

## Key Features

### 1. Word Embeddings and Semantic Search

- Word2Vec training and word-level semantic search
- Custom GloVe implementation using the original notebook algorithm
- Cosine similarity for comparing word vectors
- PCA visualization of embedding relationships
- Search ranking, input validation, and unknown-word handling

### 2. Knowledge Base Question Answering

- Editable knowledge base in Streamlit
- Document chunking and sentence embeddings
- Semantic retrieval of relevant context
- Grounded answers based on retrieved information
- Abstention when sufficient information is unavailable
- Response status, execution time, and retrieved context display

### 3. Gemini-Powered PDF AI Chat

- PDF upload and text extraction
- Semantic retrieval using Sentence Transformers
- Google Gemini integration for natural-language answers
- Multi-turn conversations and follow-up questions
- Document-grounded responses
- PDF page references and expandable document sources
- Session-based conversation history
- Graceful handling of API errors and unavailable services

### 4. Deployment and Testing

- Interactive Streamlit web application
- Deployment on Streamlit Community Cloud
- GitHub Actions continuous integration
- Python unit tests
- Environment-based API key configuration

## Live Application

**Try the deployed application:**

https://digitaltwin1.streamlit.app

The interface provides two modes:

| Mode | Description |
|---|---|
| Knowledge Base Q&A | Ask questions about an editable knowledge base using semantic retrieval and grounded answer extraction |
| PDF AI Chat | Upload a PDF and ask conversational questions using semantic retrieval and Google Gemini |

## System Architecture

```text
                       USER
                         |
                         v
                  STREAMLIT UI
                         |
             +-----------+-----------+
             |                       |
             v                       v
     KNOWLEDGE BASE Q&A          PDF AI CHAT
             |                       |
             v                       v
     Knowledge Base             PDF Upload
             |                       |
             v                       v
     Document Chunking          Text Extraction
             |                       |
             v                       v
     Sentence Embeddings        Semantic Retrieval
             |                       |
             v                       v
     Semantic Retriever         Relevant PDF Context
             |                       |
             v                       v
     Grounded Answer            Google Gemini API
       or Abstention                |
             |                       v
             |                Conversational Answer
             |                with Page References
             |                       |
             +-----------+-----------+
                         |
                         v
                  STREAMLIT RESULTS
```

The original Word2Vec and GloVe experiments remain available as a separate educational component.

See the [original architecture documentation](diagrams/architecture.md).

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Gensim | Word2Vec embeddings |
| Custom GloVe | Word embedding implementation |
| NumPy and SciPy | Numerical computation |
| Scikit-learn | Cosine similarity and PCA |
| Sentence Transformers | Semantic text embeddings |
| Google Gemini API | Generative PDF question answering |
| Streamlit | Interactive web application |
| Git and GitHub | Version control |
| GitHub Actions | Continuous integration |
| Streamlit Community Cloud | Web deployment |

## Getting Started

### Prerequisites

- Python 3.12
- Git
- Internet access for downloading embedding models and using Gemini
- Google Gemini API key for PDF AI Chat

### 1. Clone the Repository

```bash
git clone https://github.com/kefras/digital-twin-word2vec-glove-embeddings.git
cd digital-twin-word2vec-glove-embeddings
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Gemini API Access

Create a folder named `.streamlit` in the project root if it does not already exist.

Inside that folder, create a file named `secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your own API key.

**Security:** Never commit API keys or `.streamlit/secrets.toml` to GitHub. The secrets file must remain excluded by `.gitignore`.

For Streamlit Community Cloud, configure the same secret in the deployed application's settings.

### 5. Launch Digital Twin AI

From the repository root:

```bash
python -m streamlit run src/rag/app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## Using the Application

### Knowledge Base Q&A

1. Select **Knowledge Base Q&A** from the sidebar.
2. Enter or edit the knowledge base content.
3. Type a question.
4. Click **Ask Digital Twin**.
5. Review the answer, status, response time, and retrieved context.

Example questions:

- What is Docker used for?
- What does Kubernetes manage?
- Which tool automates cloud infrastructure?
- Which platform provides cloud computing services?

This mode uses semantic retrieval and grounded answer extraction rather than Gemini generation.

### PDF AI Chat

1. Select **PDF AI Chat** from the sidebar.
2. Upload a text-based PDF document.
3. Enter a question about its contents.
4. Review the generated answer and page references.
5. Ask follow-up questions within the same conversation.

Example conversation:

**User:** What does Kubernetes do?

**Digital Twin AI:** Kubernetes automates container deployment and scaling (Page 1).

**User:** How does it compare with Docker?

**Digital Twin AI:** Docker packages applications into portable containers, whereas Kubernetes automates container deployment and scaling (Page 1).

The example illustrates document-grounded answers and conversational follow-up handling.

## Original Word2Vec and GloVe Experiments

The original educational implementation remains part of the repository.

### Included Materials

- `notebooks/Week2_08_embeddings_deep_dive_completed.ipynb` — completed Colab notebook with saved outputs
- `src/glove.py` — original `SimpleGloVe` implementation extracted from the notebook
- `data/original_colab_corpus.txt` — original ten-document sample corpus
- `src/integrated_demo.py` — local Word2Vec and GloVe integration

### Run the Original Experiment

```bash
python -m src.integrated_demo
```

Additional training and search commands:

```bash
python -m src.train
python -m src.demo --query cloud --top-k 5
```

### Original Experiment Results

The completed Colab exercise reported:

| Metric | Result |
|---|---|
| Source documents | 10 |
| Preprocessed tokens | 42 |
| Word2Vec vocabulary | 29 words |
| Word2Vec vector dimensions | 50 |
| GloVe vocabulary | 29 words |
| GloVe vector dimensions | 20 |
| GloVe training epochs | 300 |
| GloVe loss | Approximately 0.6717 to 0.5162 |

These are educational experiment observations, not production benchmarks.

Trained `.model` and `.pkl` files are not included in the repository. The models can be regenerated using the supplied code.

**Security:** Do not unpickle model files obtained from untrusted sources.

## Automated Testing

Run the existing test suite:

```bash
python -m unittest discover -s tests -v
```

**Latest verified local result: 21 tests passed.**

The existing tests cover areas including:

- Word2Vec and GloVe search behavior
- Cosine similarity
- Search ranking
- Invalid input and unknown words
- Missing-model handling
- RAG retrieval behavior
- Question-answering response structure
- Abstention handling

The PDF Chat feature has also been manually tested locally and on Streamlit Cloud. Dedicated automated PDF Chat tests are a planned improvement.

## Project Structure

```text
digital-twin-embeddings/
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- data/
|   `-- original_colab_corpus.txt
|
|-- diagrams/
|   `-- architecture.md
|
|-- notebooks/
|   `-- Week2_08_embeddings_deep_dive_completed.ipynb
|
|-- src/
|   |-- rag/
|   |   |-- __init__.py
|   |   |-- app.py
|   |   |-- chunker.py
|   |   |-- pipeline.py
|   |   |-- retriever.py
|   |   |-- pdf_loader.py
|   |   |-- semantic_retriever.py
|   |   `-- pdf_chat.py
|   |
|   |-- glove.py
|   |-- integrated_demo.py
|   |-- train.py
|   `-- demo.py
|
|-- tests/
|   |-- test_original_glove.py
|   |-- test_rag.py
|   `-- test_search.py
|
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

Local test scripts, sample PDFs, secrets, virtual environments, and generated model binaries may exist during development but are not required to be committed.

## Screenshots

Screenshots of the deployed application can be added to the repository to demonstrate:

- Knowledge Base Q&A
- PDF upload
- Gemini-generated answers
- Multi-turn PDF conversations
- Document page references
- Automated test results

## Limitations

Digital Twin AI is an educational and portfolio-oriented prototype.

- Semantic retrieval quality depends on the embedding model and source content.
- PDF text extraction may not work with scanned documents without OCR.
- Gemini responses depend on API availability, model access, and usage limits.
- Generated answers should be checked against their cited source pages.
- Conversation history is limited to the active Streamlit session.
- The original small Word2Vec and GloVe corpus is not suitable for production semantic search evaluation.
- The application is not a real-world physical digital twin or a production document-management system.

## Future Improvements

- Automated tests for PDF extraction, semantic retrieval, and Gemini error handling
- Support for multiple PDF documents
- Improved document chunking and retrieval evaluation
- Persistent conversation history
- Vector database integration
- Document preview with highlighted source passages
- Performance benchmarking
- Optional OCR support
- Authentication and access controls
- Integration with domain-specific Digital Twin systems

## Project Background

Originally developed for a Week 2 AI School embeddings challenge, this project demonstrates a progression from traditional word embeddings to practical semantic retrieval and AI-assisted document question answering.

It serves as a hands-on learning project in machine learning, natural language processing, Retrieval-Augmented Generation, generative AI integration, and cloud application deployment.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).