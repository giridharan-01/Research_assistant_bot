import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="JIM - PDF Assistant", layout="wide")

st.write("API Key Loaded:", bool(os.getenv("OPENAI_API_KEY")))

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
# QUICK STATS (SESSION BASED)
# =========================
st.subheader("📊 Session Summary")

col1, col2, col3 = st.columns(3)

col1.metric("💬 Messages", len(st.session_state.get("messages", [])))
col2.metric("📘 Units Loaded", len(st.session_state.get("question_bank", {})))
col3.metric("✅ Selected Questions",
            sum(len(v) for v in st.session_state.get("selected_questions", {}).values()))

st.markdown("---")

# =========================
# CTA
# =========================
st.success("👉 Use the sidebar to start exploring the app!")
