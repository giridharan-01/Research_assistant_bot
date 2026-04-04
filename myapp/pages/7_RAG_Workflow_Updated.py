import streamlit as st
import numpy as np

st.title("🔍 RAG Workflow Demo (Using Existing FAISS DB)")

# ---------------- CHECK ---------------- #
if "db" not in st.session_state:
    st.error("❌ No vector DB found. Please upload and process PDF in previous tab.")
    st.stop()

db = st.session_state.db
retriever = st.session_state.retriever
chunks = st.session_state.get("chunks", None)

# ---------------- STEP 1: CHUNK COUNT ---------------- #
st.subheader("📄 Step 1: Chunk Information")

if chunks:
    st.success(f"✅ Total Chunks Created: {len(chunks)}")
else:
    st.warning("⚠️ Chunk data not available.")

# ---------------- QUERY ---------------- #
query = st.text_input("🔍 Enter your query")

if query:

    embeddings = st.session_state.embeddings

    # ---------------- STEP 2: EMBEDDING ---------------- #
    st.subheader("🧠 Step 2: Query Embedding")

    query_embedding = embeddings.embed_query(query)
    st.write(f"Embedding Dimension: {len(query_embedding)}")

    # ---------------- STEP 3: SIMILARITY SEARCH ---------------- #
    st.subheader("🔍 Step 3: Similarity Search")

    docs_and_scores = db.similarity_search_with_score(query, k=5)

    retrieved_docs = []
    chunk_mapping = {}

    st.write("### Top Matching Chunks:")

    for i, (doc, score) in enumerate(docs_and_scores):

        similarity = 1 / (1 + score)

        # find chunk index
        chunk_index = None
        if chunks:
            for idx, c in enumerate(chunks):
                if c.page_content == doc.page_content:
                    chunk_index = idx
                    break

        label = f"Chunk {chunk_index} | Similarity: {similarity:.4f}" if chunk_index is not None else f"Match {i+1}"

        st.write(label)

        if chunk_index is not None:
            chunk_mapping[chunk_index] = doc.page_content

        retrieved_docs.append(doc.page_content)

    # ---------------- STEP 4: DROPDOWN INSPECTION ---------------- #
    st.subheader("🧪 Step 4: Inspect Retrieved Chunk")

    if chunk_mapping:
        selected_chunk = st.selectbox(
            "Select a chunk to inspect",
            options=list(chunk_mapping.keys())
        )

        st.write(f"### Content of Chunk {selected_chunk}")
        st.info(chunk_mapping[selected_chunk])

    # ---------------- STEP 5: CONTEXT ---------------- #
    st.subheader("📚 Step 5: Context Sent to LLM")

    context = "\n\n".join(retrieved_docs)
    st.code(context[:2000])

    # ---------------- STEP 6: FINAL ANSWER ---------------- #
    st.subheader("💬 Step 6: Final Answer")

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
