import streamlit as st
import numpy as np

st.title("🔍 RAG Workflow Demo (Using Existing FAISS DB)")

# ---------------- CHECK ---------------- #
if "db" not in st.session_state:
    st.error("❌ No vector DB found. Please upload and process PDF in previous tab.")
    st.stop()

db = st.session_state.db
retriever = st.session_state.retriever

# Optional (if you stored chunks earlier)
chunks = st.session_state.get("chunks", None)

# ---------------- SHOW CHUNKS ---------------- #
st.subheader("📄 Step 1: Document Chunks")

if chunks:
    for i, chunk in enumerate(chunks[:10]):
        with st.expander(f"Chunk {i+1}"):
            st.write(chunk.page_content)
else:
    st.warning("⚠️ Chunks not stored earlier. Only similarity demo will be shown.")

# ---------------- QUERY ---------------- #
query = st.text_input("🔍 Enter your query")

if query:

    embeddings = st.session_state.embeddings

    # ---------------- QUERY EMBEDDING ---------------- #
    st.subheader("🧠 Step 2: Query Embedding")

    query_embedding = embeddings.embed_query(query)
    st.write(f"Embedding Dimension: {len(query_embedding)}")

    # ---------------- SIMILARITY SEARCH ---------------- #
    st.subheader("🔍 Step 3: Similarity Search")

    docs_and_scores = db.similarity_search_with_score(query, k=5)

    retrieved_docs = []

    for i, (doc, score) in enumerate(docs_and_scores):

        similarity = 1 / (1 + score)

        with st.expander(f"Match {i+1} | Similarity: {similarity:.4f}"):
            st.write(doc.page_content)

        retrieved_docs.append(doc.page_content)

    # ---------------- CONTEXT ---------------- #
    st.subheader("📚 Step 4: Context Sent to LLM")

    context = "\n\n".join(retrieved_docs)
    st.code(context[:2000])

    # ---------------- FINAL ANSWER ---------------- #
    st.subheader("💬 Step 5: Final Answer")

    llm = st.session_state.llm

    response = llm.invoke(f"""
    Answer ONLY using the context below.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """)

    st.success(response.content)
