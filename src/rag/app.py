
import streamlit as st

from src.rag.pipeline import DigitalTwinRAG


st.set_page_config(
    page_title="Digital Twin AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Digital Twin AI")
st.caption("Retrieval-Augmented Generation | Semantic Search")


DEFAULT_DOCUMENT = """
AWS provides cloud computing services.
Docker helps developers build and deploy containers.
Kubernetes manages containerized applications.
Terraform automates cloud infrastructure deployment.
"""


@st.cache_resource
def load_rag(document):
    return DigitalTwinRAG(document)


with st.sidebar:
    st.header("Knowledge Base")

    document = st.text_area(
        "Knowledge base content",
        value=DEFAULT_DOCUMENT,
        height=250
    )

    st.info("Edit the knowledge base to explore different topics.")


if not document.strip():
    st.warning("Please enter knowledge base content.")
    st.stop()


try:
    rag = load_rag(document)
except Exception as error:
    st.error(f"Unable to initialize the RAG system: {error}")
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

            col1.metric("Status", result["status"])
            col2.metric(
                "Response Time",
                f'{result["duration_seconds"]:.4f}s'
            )

            with st.expander("Retrieved Context"):
                for chunk in result["retrieved_chunks"]:
                    st.write(chunk)

        except Exception as error:
            st.error(f"Question processing failed: {error}")
