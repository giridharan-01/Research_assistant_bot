import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="RAG Workflow Demo", layout="wide")
st.title("🔍 RAG Workflow Demo (Advanced Visualization)")

# ---------------- CHECK ---------------- #
if "db" not in st.session_state:
    st.error("❌ No vector DB found. Please upload and process PDF in previous tab.")
    st.stop()

db = st.session_state.db
chunks = st.session_state.get("chunks", None)
embeddings = st.session_state.embeddings
llm = st.session_state.llm

# ---------------- STEP 1: CHUNK INFO ---------------- #
st.subheader("📄 Step 1: Chunk Information")

if chunks:
    st.success(f"✅ Total Chunks Created: {len(chunks)}")
else:
    st.warning("⚠️ Chunk data not available.")

# ---------------- QUERY ---------------- #
query = st.text_input("🔍 Enter your query")

# -------- FUNCTION: HIGHLIGHT -------- #
def highlight_text(text, query):
    words = query.lower().split()
    for word in words:
        text = text.replace(word, f"**:red[{word}]**")
        text = text.replace(word.capitalize(), f"**:red[{word.capitalize()}]**")
    return text

if query:

    # ---------------- STEP 2: EMBEDDING ---------------- #
    st.subheader("🧠 Step 2: Query Embedding")

    query_embedding = np.array(embeddings.embed_query(query))
    st.write(f"Embedding Dimension: {len(query_embedding)}")

    # ---------------- STEP 3: SIMILARITY SEARCH ---------------- #
    st.subheader("🔍 Step 3: Similarity Search")

    docs_and_scores = db.similarity_search_with_score(query, k=5)

    retrieved_docs = []
    chunk_indices = []
    similarities = []
    chunk_mapping = {}

    for i, (doc, score) in enumerate(docs_and_scores):

        similarity = 1 / (1 + score)

        # Find chunk index
        chunk_index = None
        if chunks:
            for idx, c in enumerate(chunks):
                if c.page_content == doc.page_content:
                    chunk_index = idx
                    break

        if chunk_index is not None:
            chunk_indices.append(chunk_index)
            similarities.append(similarity)
            chunk_mapping[chunk_index] = doc.page_content

        retrieved_docs.append(doc.page_content)

        st.write(f"Chunk {chunk_index} → Similarity: {similarity:.4f}")

    # ---------------- STEP 4: VISUALIZATION ---------------- #
# ---------------- STEP 4: VISUALIZATION ---------------- #
    st.subheader("📊 Step 4: Top-5 Chunk Similarity (Ranked)")

    if chunk_indices:

        df = pd.DataFrame({
            "Chunk Index": chunk_indices,
            "Similarity": similarities
        })

        # Sort by similarity (descending)
        df = df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)

        # ✅ Add Rank column
        df["Rank"] = df.index + 1

        # Convert chunk index to string
        df["Chunk Index"] = df["Chunk Index"].astype(str)

        # Create label like "Top 1 (Chunk 23)"
        df["Label"] = df.apply(lambda x: f"Top {x['Rank']} (Chunk {x['Chunk Index']})", axis=1)

        # 🎯 Plot
        fig = px.bar(
            df,
            x="Label",
            y="Similarity",
            color="Similarity",
            color_continuous_scale="RdYlGn",
            title="📊 Top-5 Retrieved Chunks (Ranked by Similarity)",
            text_auto=True
        )

        fig.update_layout(
            xaxis_title="Top-K Chunks",
            yaxis_title="Similarity Score"
        )

        st.plotly_chart(fig, use_container_width=True)

#    st.subheader("📊 Step 4: Similarity Visualization (Ranked)")
#
#    if chunk_indices:
#
#        df = pd.DataFrame({
#            "Chunk Index": chunk_indices,
#            "Similarity": similarities
#        })
#
#        # Sort by similarity
#        df = df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)
#
#        df["Chunk Index"] = df["Chunk Index"].astype(str)
#
#        fig = px.bar(
#            df,
#            x="Chunk Index",
#            y="Similarity",
#            color="Similarity",
#            color_continuous_scale="RdYlGn",
#            title="📊 Ranked Chunk Similarity",
#            text_auto=True
#        )
#
#        fig.update_layout(
#            xaxis_title="Chunk Index (Ranked)",
#            yaxis_title="Similarity Score"
#        )
#
#        st.plotly_chart(fig, use_container_width=True)

        # ---------------- STEP 5: TOP CHUNK ---------------- #
        st.subheader("🏆 Step 5: Top Matching Chunk")

        top_chunk_index = int(df.iloc[0]["Chunk Index"])
        top_chunk_text = chunk_mapping.get(top_chunk_index, "")

        st.success(f"Top Chunk: {top_chunk_index} (Highest Similarity)")

        highlighted_top = highlight_text(top_chunk_text, query)
        st.markdown(highlighted_top)

        # ---------------- STEP 6: DROPDOWN ---------------- #
        st.subheader("🧪 Step 6: Inspect Other Chunks")

        selected_chunk = st.selectbox(
            "Select a chunk to inspect",
            options=df["Chunk Index"].astype(int).tolist()
        )

        selected_text = chunk_mapping.get(selected_chunk, "")
        highlighted_selected = highlight_text(selected_text, query)

        st.markdown(f"### Chunk {selected_chunk}")
        st.markdown(highlighted_selected)

    # ---------------- STEP 7: CONTEXT ---------------- #
    st.subheader("📚 Step 7: Context Sent to LLM")

    context = "\n\n".join(retrieved_docs)
    st.code(context[:2000])

    # ---------------- STEP 8: FINAL ANSWER ---------------- #
    st.subheader("💬 Step 8: Final Answer")

    response = llm.invoke(f"""
    Answer ONLY using the context below.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """)

    st.success(response.content)
