import streamlit as st

st.set_page_config(page_title="JIM - PDF Assistant", layout="wide")

# =========================
# HEADER
# =========================
st.title("🤖 JIM - PDF Assistant")
st.markdown("### 🚀 Your Smart Academic Companion - JIM")

st.markdown("---")

# =========================
# HERO SECTION
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 👋 Welcome!

    This application helps you:

    - 📄 Chat with PDF documents  
    - 📘 Generate exam questions automatically  
    - 📥 Export chats and question papers  

    💡 Built using **LLMs + RAG + Streamlit**
    """)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=200)

st.markdown("---")

# =========================
# FEATURES SECTION
# =========================
st.subheader("✨ Features")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("### 📄 PDF Assistant")
    st.markdown("""
    - Upload multiple PDFs  
    - Ask questions  
    - Context-aware answers  
    """)

with f2:
    st.markdown("### ⬇️ Download Chat")
    st.markdown("""
    - Export chat history  
    - Clean formatted PDF  
    - Ready for documentation  
    """)

with f3:
    st.markdown("### 📘 Question Generator")
    st.markdown("""
    - Generate exam questions  
    - Based on syllabus  
    - Bloom’s taxonomy aligned  
    """)

st.markdown("---")

# =========================
# HOW TO USE
# =========================
st.subheader("🧭 How to Use")

st.markdown("""
1. 👉 Go to **PDF Assistant** → Upload and chat  
2. 👉 Go to **Question Generator** → Upload syllabus  
3. 👉 Select questions → Export  
4. 👉 Download results from **Download Chat**
""")

st.markdown("---")

# =========================
# SESSION STATS
# =========================
st.subheader("📊 Session Summary")

col1, col2, col3 = st.columns(3)

col1.metric("💬 Messages", len(st.session_state.get("messages", [])))
col2.metric("📘 Units Loaded", len(st.session_state.get("question_bank", {})))
col3.metric("✅ Selected Questions",
            sum(len(v) for v in st.session_state.get("selected_questions", {}).values()))

st.markdown("---")

# =========================
# RAG EXPLANATION (ADVANCED)
# =========================
st.subheader("🧠 How Retrieval Works (FAISS + Similarity)")

with st.expander("🔍 Vector Search using FAISS", expanded=True):
    st.markdown("""
FAISS (Facebook AI Similarity Search) is used to efficiently retrieve the most relevant document chunks.

Instead of keyword matching, we use **semantic similarity** via embeddings.
""")

with st.expander("📐 L2 Distance (Euclidean Distance)"):
    st.latex(r"d(q, x) = \sqrt{\sum_{i=1}^{n} (q_i - x_i)^2}")

    st.markdown("""
- **q** → Query embedding  
- **x** → Document embedding  

👉 Smaller distance = More similar
""")

with st.expander("🔄 Similarity Conversion"):
    st.latex(r"Similarity = \frac{1}{1 + d(q, x)}")

    st.markdown("""
- Converts distance → similarity score  
- Value closer to **1 = highly relevant**
""")

# =========================
# COSINE vs L2 COMPARISON
# =========================
st.subheader("⚖️ Cosine vs L2 Similarity")

st.table({
    "Metric": ["L2 Distance", "Cosine Similarity"],
    "Definition": [
        "Measures absolute distance between vectors",
        "Measures angle between vectors"
    ],
    "Range": [
        "0 → ∞",
        "-1 → 1"
    ],
    "Interpretation": [
        "Lower = more similar",
        "Higher = more similar"
    ],
    "Used in this App": [
        "✅ Yes (FAISS default)",
        "❌ Not used (but possible)"
    ]
})

st.markdown("---")

# =========================
# RAG PIPELINE VISUAL
# =========================
st.subheader("📊 RAG Pipeline")

st.markdown("""
```text
User Query
    ↓
Convert to Embedding
    ↓
FAISS Vector Search (L2 Distance)
    ↓
Top-K Relevant Chunks
    ↓
Context Creation
    ↓
LLM (Answer Generation)
    ↓
Final Response""")

st.info("💡 This pipeline ensures accurate, context-aware, and hallucination-free responses.")

st.markdown("---")

st.success("👉 Use the sidebar to start exploring the app!")
