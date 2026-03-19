# =========================
# PART 1: IMPORTS & SETUP
# =========================

import streamlit as st
import os
import shutil
import datetime
import markdown
import weasyprint
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from dotenv import load_dotenv
from PyPDF2 import PdfReader

# ===== LangChain (Latest Compatible Imports) =====
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain
# ================================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


load_dotenv()

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📚",
    layout="wide"
)

st.title("Hi, I am Ray..")
st.markdown("Your **PDF Assistant**")

# Chat history store
store = {}

# =========================
# FILE HANDLING UTILITIES
# =========================

def save_uploaded_file(upload_file):
    os.makedirs("tempDir", exist_ok=True)
    with open(os.path.join("tempDir", upload_file.name), "wb") as f:
        f.write(upload_file.getbuffer())

def delete_contents():
    folder = os.path.join(os.getcwd(), "tempDir")
    if not os.path.exists(folder):
        return
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            else:
                shutil.rmtree(path)
        except Exception as e:
            print(f"Delete failed: {e}")

def extract_metadata(file_path):
    reader = PdfReader(file_path)
    info = reader.metadata
    title = info.title if info and info.title else "Unknown Title"
    authors = info.author if info and info.author else "Unknown Authors"
    return title, authors

# =========================
# FAISS + RAG INITIALIZATION
# =========================

def initialize_setup(doc_pages):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(doc_pages, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return llm, retriever

def create_rag_chain(llm, retriever):
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and user question, rewrite it as a standalone question. "
         "Do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Use the following context to answer the question. "
         "If you don't know, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_rag_pipeline(rag_chain):
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def get_response(rag_obj, query):
    result = rag_obj.invoke(
        {"input": query},
        config={"configurable": {"session_id": "user123"}}
    )
    return result["answer"]

# =========================
# PDF GENERATION
# =========================

def create_pdf(details_str, file_name):
    html_content = markdown.markdown(details_str)
    html = f"""
    <html>
    <body style="font-family:sans-serif">
        {html_content}
    </body>
    </html>
    """
    weasyprint.HTML(string=html).write_pdf(file_name)

# =========================
# PART 2: STREAMLIT UI
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    # st.session_state.llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     api_key=os.getenv("OPENAI_API_KEY")
    # )
    # st.session_state.llm = ChatOpenAI(
    # model="gpt-4o-mini",
    # openai_api_key=os.getenv("OPENAI_API_KEY")
    # )
    st.session_state.llm = ChatOpenAI(
    model="gpt-4o-mini",
    client=client   # ✅ THIS IS THE FIX
    )

t1, t2 = st.tabs(["📄 PDF Assistant", "💡 Project Idea Generator"])

# =========================
# TAB 1: PDF ASSISTANT
# =========================

with t1:
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and "rag_obj" not in st.session_state:
        doc_pages = []
        metadata = []

        for file in uploaded_files:
            st.success(f"Processing {file.name}")
            save_uploaded_file(file)

            file_path = os.path.join("tempDir", file.name)
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            doc_pages.extend(pages)

            title, authors = extract_metadata(file_path)
            metadata.append({
                "title": title,
                "authors": authors,
                "pages": len(pages)
            })

        delete_contents()

        llm, retriever = initialize_setup(doc_pages)
        rag_chain = create_rag_chain(llm, retriever)

        st.session_state.rag_obj = create_rag_pipeline(rag_chain)
        st.session_state.metadata = metadata

    if "rag_obj" in st.session_state:
        for meta in st.session_state.metadata:
            st.write(f"**Title:** {meta['title']}")
            st.write(f"**Authors:** {meta['authors']}")
            st.write(f"**Pages:** {meta['pages']}")
            st.divider()

        if query := st.chat_input("Ask a question about the PDFs"):
            with st.chat_message("user"):
                st.markdown(query)

            response = get_response(st.session_state.rag_obj, query)

            with st.chat_message("assistant"):
                st.markdown(response)

# =========================
# TAB 2: PROJECT IDEAS
# =========================

with t2:
    st.title("Project Idea Generator")

    degree_level = st.selectbox(
        "Select Degree Level",
        ["Bachelor's", "Master's", "PhD", "Professional Certification"]
    )

    technology = st.text_area(
        "Technology / Domain",
        placeholder="Enter your area of interest"
    )

    if st.button("Generate Project Ideas"):
        prompt = (
            f"Generate 5 unique project titles for a "
            f"{degree_level} student using {technology}. "
            f"Return only titles."
        )

        response = st.session_state.llm.invoke(prompt)
        st.markdown(response.content)

