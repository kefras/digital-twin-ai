
import sys
import hashlib
from pathlib import Path

import streamlit as st

# Ensure the project root is available on Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import DigitalTwinRAG
from src.rag.pdf_chat import ask_pdf


DEFAULT_DOCUMENT = """
AWS provides cloud computing services.
Docker helps developers build and deploy containers.
Kubernetes manages containerized applications.
Terraform automates cloud infrastructure deployment.
"""


st.set_page_config(
    page_title="Digital Twin AI Assistant",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_rag(document):
    return DigitalTwinRAG(document)


def get_gemini_api_key():
    """Read the Gemini API key from Streamlit secrets."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


# Initialize session state
if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []

if "pdf_fingerprint" not in st.session_state:
    st.session_state.pdf_fingerprint = None


st.title("🤖 Digital Twin AI Assistant")
st.caption(
    "Explore knowledge bases and chat with PDF documents "
    "using Retrieval-Augmented Generation and Google Gemini."
)


mode = st.sidebar.radio(
    "Select AI Mode",
    ["Knowledge Base Q&A", "PDF AI Chat"]
)


# -------------------------------------------------
# MODE 1: ORIGINAL KNOWLEDGE BASE RAG
# -------------------------------------------------

if mode == "Knowledge Base Q&A":

    with st.sidebar:
        st.header("Knowledge Base")

        document = st.text_area(
            "Knowledge base content",
            value=DEFAULT_DOCUMENT,
            height=250
        )

        st.info(
            "Edit the knowledge base to explore different topics."
        )

    st.header("Knowledge Base Question Answering")

    if not document.strip():
        st.warning("Please enter knowledge base content.")
        st.stop()

    try:
        rag = load_rag(document)

    except Exception as error:
        st.error(
            f"Unable to initialize the RAG system: {error}"
        )
        st.stop()

    question = st.text_input(
        "Ask your Digital Twin a question",
        placeholder="What is Docker used for?"
    )

    if st.button("Ask Digital Twin", type="primary"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Searching knowledge base..."):
                    result = rag.ask(question)

                st.subheader("Answer")
                st.write(result["answer"])

                col1, col2 = st.columns(2)

                col1.metric(
                    "Status",
                    result["status"]
                )

                col2.metric(
                    "Response Time",
                    f'{result["duration_seconds"]:.4f}s'
                )

                with st.expander("Retrieved Context"):
                    for chunk in result["retrieved_chunks"]:
                        st.write(chunk)

            except Exception as error:
                st.error(
                    f"Question processing failed: {error}"
                )


# -------------------------------------------------
# MODE 2: GEMINI PDF AI CHAT
# -------------------------------------------------

elif mode == "PDF AI Chat":

    st.header("Chat With Your PDF Documents")

    st.write(
        "Upload a PDF document and ask questions about its content. "
        "Google Gemini generates answers using retrieved document passages."
    )

    api_key = get_gemini_api_key()

    if not api_key:
        st.error(
            "Gemini API key not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        accept_multiple_files=False
    )

    if uploaded_file is None:
        st.info("Upload a PDF to begin chatting.")
        st.stop()

    pdf_bytes = uploaded_file.getvalue()

    fingerprint = hashlib.sha256(pdf_bytes).hexdigest()

    # Reset chat when a different PDF is uploaded
    if st.session_state.pdf_fingerprint != fingerprint:
        st.session_state.pdf_messages = []
        st.session_state.pdf_fingerprint = fingerprint

    st.success(
        f"Document uploaded: {uploaded_file.name}"
    )

    st.caption(
        f"Document size: {len(pdf_bytes) / 1024:.1f} KB"
    )

    with st.sidebar:
        st.header("PDF Chat Settings")

        st.write(f"Current document: {uploaded_file.name}")

        if st.button("Clear Chat History"):
            st.session_state.pdf_messages = []
            st.rerun()

        st.info(
            "Conversation history is stored only in this "
            "Streamlit browser session."
        )

    # Display previous messages
    for message in st.session_state.pdf_messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Document Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"{source['filename']} — "
                            f"Page {source['page']}"
                        )

    # Accept new questions
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.pdf_messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Gemini is analyzing your PDF..."):

                try:
                    result = ask_pdf(
    pdf_bytes=pdf_bytes,
    filename=uploaded_file.name,
    question=question,
    api_key=api_key,
    history=st.session_state.pdf_messages[:-1]
)

                    answer = result["answer"]
                    sources = result.get("sources", [])

                except Exception:
                    answer = (
                        "Sorry, I couldn't process your question. "
                        "Please try again."
                    )
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander("Document Sources"):
                    for source in sources:
                        st.write(
                            f"{source['filename']} — "
                            f"Page {source['page']}"
                        )

        st.session_state.pdf_messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
