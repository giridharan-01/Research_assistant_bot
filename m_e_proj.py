# =========================
# PART 1: IMPORTS & SETUP
# =========================

import streamlit as st
import os
import shutil
import markdown
import weasyprint

# 🔥 FIX PROXY ISSUE
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("Hi, I am Ray..")
st.markdown("Your **PDF Assistant**")

# =========================
# SESSION STATE
# =========================

store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# =========================
# FILE HANDLING
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
        except:
            pass

def extract_metadata(file_path):
    reader = PdfReader(file_path)
    info = reader.metadata
    return (
        info.title if info and info.title else "Unknown Title",
        info.author if info and info.author else "Unknown Authors"
    )

# =========================
# RAG SETUP
# =========================

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def initialize_setup(doc_pages):
    llm = get_llm()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(doc_pages, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return llm, retriever

def create_rag_chain(llm, retriever):
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question standalone. Do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use context to answer. If unknown, say I don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)

def get_session_history(session_id):
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

def create_pdf(text, filename):
    html = markdown.markdown(text)
    weasyprint.HTML(string=html).write_pdf(filename)

# =========================
# UI
# =========================

t1, t2 = st.tabs(["📄 PDF Assistant", "💡 Project Ideas"])

# =========================
# TAB 1: PDF CHAT
# =========================

with t1:
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and "rag_obj" not in st.session_state:
        doc_pages = []
        metadata = []

        for file in uploaded_files:
            save_uploaded_file(file)

            path = os.path.join("tempDir", file.name)
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            doc_pages.extend(pages)

            title, author = extract_metadata(path)
            metadata.append({"title": title, "author": author})

        delete_contents()

        llm, retriever = initialize_setup(doc_pages)
        rag_chain = create_rag_chain(llm, retriever)

        st.session_state.rag_obj = create_rag_pipeline(rag_chain)
        st.session_state.metadata = metadata

    if "rag_obj" in st.session_state:

        # 🔥 SHOW METADATA
        for m in st.session_state.metadata:
            st.write(f"**Title:** {m['title']}")
            st.write(f"**Author:** {m['author']}")
            st.divider()

        # 🔥 SHOW CHAT HISTORY
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 🔥 CHAT INPUT
        if query := st.chat_input("Ask your question"):

            # Save user message
            st.session_state.messages.append({
                "role": "user",
                "content": query
            })

            with st.chat_message("user"):
                st.markdown(query)

            # Get response
            response = get_response(st.session_state.rag_obj, query)

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            with st.chat_message("assistant"):
                st.markdown(response)

        # 🔥 CLEAR CHAT BUTTON
        if st.button("Clear Chat"):
            st.session_state.messages = []
            store.clear()

# =========================
# TAB 2: PROJECT IDEAS
# =========================

with t2:
    st.subheader("Project Idea Generator")

    degree = st.selectbox(
        "Degree",
        ["Bachelor's", "Master's", "PhD"]
    )

    tech = st.text_area("Technology / Domain")

    if st.button("Generate"):
        prompt = f"Generate 5 project ideas for {degree} in {tech}"

        response = st.session_state.llm.invoke(prompt)

        st.markdown(response.content)
