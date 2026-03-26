# =========================
# IMPORTS & SETUP
# =========================

import streamlit as st
import os
import shutil
import markdown
import weasyprint
import json
import re

os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("Hi, I am Ray..")
st.markdown("Your **PDF Assistant**")

# =========================
# SESSION
# =========================

store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = {"3M": [], "5M": [], "10M": []}

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")

# =========================
# FILE
# =========================

def save_uploaded_file(upload_file):
    os.makedirs("tempDir", exist_ok=True)
    with open(os.path.join("tempDir", upload_file.name), "wb") as f:
        f.write(upload_file.getbuffer())

# =========================
# RAG
# =========================

def initialize_setup(doc_pages):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(doc_pages, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# =========================
# PDF
# =========================

def create_pdf(text, filename):
    html = markdown.markdown(text)
    weasyprint.HTML(string=html).write_pdf(filename)

# =========================
# TABS
# =========================

t1, t2 = st.tabs(["📄 PDF Assistant", "📘 Question Generator"])

# =========================
# TAB 1 (FIXED CHAT INPUT)
# =========================

with t1:

    st.markdown("""
    <style>
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
    }
    .input-box {
        position: fixed;
        bottom: 0;
        width: 70%;
        background: white;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files and "retriever" not in st.session_state:
        docs = []
        for file in uploaded_files:
            save_uploaded_file(file)
            loader = PyPDFLoader(os.path.join("tempDir", file.name))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            docs.extend(splitter.split_documents(pages))

        st.session_state.retriever = initialize_setup(docs)

    if "retriever" in st.session_state:

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])

        st.markdown('</div>', unsafe_allow_html=True)

        query = st.chat_input("Ask your question")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            docs = st.session_state.retriever.get_relevant_documents(query)
            context = "\n".join([d.page_content for d in docs])

            response = st.session_state.llm.invoke(f"Answer:\n{context}\nQ:{query}")

            st.session_state.messages.append({"role": "assistant", "content": response.content})
            st.rerun()

# =========================
# TAB 2 (CHECKBOX SYSTEM)
# =========================

with t2:

    uploaded_pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")

    if "question_bank" not in st.session_state:
        st.session_state.question_bank = {}

    def extract_text(file):
        reader = PdfReader(file)
        return "\n".join([p.extract_text() for p in reader.pages])

    if uploaded_pdf and st.button("Generate Questions"):

        syllabus = extract_text(uploaded_pdf)

        prompt = f"""
        Generate MANY questions per unit.
        Return JSON format.
        Each unit must have at least:
        - 15 questions (3M)
        - 10 questions (5M)
        - 8 questions (10M)
        Syllabus:\n{syllabus}
        """

        res = st.session_state.llm.invoke(prompt)
        match = re.search(r"\{.*\}", res.content, re.DOTALL)

        if match:
            st.session_state.question_bank = json.loads(match.group())

    # LIMITS
    limits = {"3M": 10, "5M": 5, "10M": 4}

    for mark_type in ["3M", "5M", "10M"]:

        st.subheader(f"{mark_type} Questions")
        selected = st.session_state.selected_questions[mark_type]

        for unit, data in st.session_state.question_bank.items():
            st.markdown(f"### {unit}")

            for i, q in enumerate(data.get(mark_type, [])):
                key = f"{unit}_{mark_type}_{i}"
                checked = st.checkbox(q, key=key)

                if checked and q not in selected:
                    if len(selected) < limits[mark_type]:
                        selected.append(q)
                    else:
                        st.warning(f"Max {limits[mark_type]} reached")

                elif not checked and q in selected:
                    selected.remove(q)

    st.divider()

    if st.button("📄 Export Selected Questions"):
        text = ""
        for mtype, qs in st.session_state.selected_questions.items():
            text += f"## {mtype}\n"
            for q in qs:
                text += f"- {q}\n"

        create_pdf(text, "selected_questions.pdf")

        with open("selected_questions.pdf", "rb") as f:
            st.download_button("Download", f, "selected_questions.pdf")
