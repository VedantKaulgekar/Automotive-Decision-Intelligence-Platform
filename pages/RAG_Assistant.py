import streamlit as st
from modules.rag_engine import query_rag
from modules.pdf_generator import generate_rag_report
import os

# Auto-detect cloud vs local
USE_GROQ = "GROQ_API_KEY" in os.environ

if USE_GROQ:
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
else:
    from langchain_community.llms import Ollama
    ollama_client = Ollama(model="mistral", temperature=0)


st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🤖 RAG Assistant")

# -------------------------------------
# SESSION STATE INIT
# -------------------------------------
if "answer" not in st.session_state:
    st.session_state.answer = None
if "sources" not in st.session_state:
    st.session_state.sources = None
if "query_done" not in st.session_state:
    st.session_state.query_done = False
if "loading" not in st.session_state:
    st.session_state.loading = False

# -------------------------------------
# INPUT BAR
# -------------------------------------
query = st.text_input("Ask your question…", value="", key="query_box")

if st.button("Search"):
    st.session_state.loading = True
    st.session_state.query_done = True
    st.session_state.answer = None
    st.session_state.sources = None

    with st.spinner("📚 Searching knowledge base…"):
        status, chunks = query_rag(query)

    # --------------------------
    # HANDLE EMPTY STATES
    # --------------------------
    if status == "NO_INDEX":
        st.warning("⚠ Knowledge base empty. Please upload documents & rebuild index.")
        st.stop()

    if status == "NO_RELEVANT":
        st.warning("⚠ No relevant paragraphs found for this question.")
        st.stop()

    if not chunks:
        st.warning("⚠ No matching text found in documents.")
        st.stop()

    # ------------------------------------------
    # SAFE NORMALIZATION OF CHUNK STRUCTURE
    # ------------------------------------------
    def safe_chunk(c):
        return {
            "source": c.get("source", "Unknown.pdf"),
            "paragraph": c.get("paragraph", c.get("text", "")),
            "score": float(c.get("score", 0.0)),
        }

    clean_chunks = [safe_chunk(c) for c in chunks]

    # ------------------------------------------
    # BUILD CONTEXT FOR THE LLM
    # ------------------------------------------
    context = "\n\n".join(
        f"[SOURCE: {c['source']} | SCORE: {c['score']:.2f}]\n{c['paragraph']}"
        for c in clean_chunks
    )

    # ------------------------------------------
    # LLM CALL
    # ------------------------------------------
    with st.spinner("🤖 AI is thinking…"):
        prompt = f"""
You are an automotive compliance assistant.

You must answer STRICTLY based on the context snippets below.
If the context does NOT contain the answer, reply:
"❗ The knowledge base does not contain information about this."

Each snippet is prefixed with its source. DO NOT use outside knowledge.

---------------- CONTEXT SNIPPETS ----------------
{context}
--------------------------------------------------

User Question: {query}

Provide:
1. A precise answer using only the snippets.
2. Cite the snippet sources you used.
"""
    with st.spinner("🤖 Generating answer…"):

        if USE_GROQ:
            # Cloud → Groq LLaMA-3 70B
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            ans = response.choices[0].message.content

        else:
            # Localhost → Ollama Mistral
            ans = ollama_client.invoke(prompt)


    st.session_state.answer = ans
    st.session_state.sources = clean_chunks
    st.session_state.loading = False

# -------------------------------------
# SHOW RESULTS
# -------------------------------------
if st.session_state.query_done and st.session_state.answer:

    st.write("### 📘 Answer")
    st.write(st.session_state.answer)

    st.write("### 📄 Sources (Matched Excerpts)")

    for s in st.session_state.sources:
        st.markdown(f"**📌 Source File:** `{s['source']}`")
        st.markdown(f"**Relevance Score:** `{s['score']:.2f}`")
        st.markdown(f"> {s['paragraph']}")
        st.markdown("---")

    # ------------------------------
    # PDF Export
    # ------------------------------
    pdf = generate_rag_report(
        query=query,
        answer=st.session_state.answer,
        sources=st.session_state.sources
    )

    st.download_button(
        "📄 Download RAG Report",
        data=pdf,
        file_name="rag_report.pdf",
        mime="application/pdf",
    )
