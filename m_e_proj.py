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

# Fix proxy issues
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain
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

# UI Styling
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
        ("system", "Answer using context and mention source page numbers. If unknown, say I don't know.\n\n{context}"),
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

t1, t2 = st.tabs(["📄 PDF Assistant", "💡 Question Generator"])

# =========================
# TAB 1
# =========================

with t1:
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("🔄 Upload New PDF"):
        st.session_state.pop("rag_obj", None)
        st.session_state.pop("metadata", None)
        st.session_state.messages = []
        store.clear()
        st.rerun()

    if uploaded_files and "rag_obj" not in st.session_state:
        doc_pages = []
        metadata = []

        for file in uploaded_files:
            save_uploaded_file(file)
            path = os.path.join("tempDir", file.name)

            loader = PyPDFLoader(path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            pages = splitter.split_documents(pages)

            doc_pages.extend(pages)

            title, author = extract_metadata(path)
            metadata.append({"title": title, "author": author})

        delete_contents()

        llm, retriever = initialize_setup(doc_pages)
        rag_chain = create_rag_chain(llm, retriever)

        st.session_state.rag_obj = create_rag_pipeline(rag_chain)
        st.session_state.metadata = metadata

    if "rag_obj" in st.session_state:

        for m in st.session_state.metadata:
            st.write(f"**Title:** {m['title']}")
            st.write(f"**Author:** {m['author']}")
            st.divider()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask your question")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            response = get_response(st.session_state.rag_obj, query)

            st.session_state.messages.append({"role": "assistant", "content": response})

            st.rerun()

        if st.button("Clear Chat"):
            st.session_state.messages = []
            store.clear()
            st.rerun()

        if st.button("📥 Download Chat as PDF"):
            chat_text = ""
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_text += f"**{role}:** {msg['content']}\n\n"

            create_pdf(chat_text, "chat.pdf")

            with open("chat.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name="chat.pdf")

# =========================
# TAB 2
# =========================

with t2:
    st.title("📘 Question Generator from PDF")

    uploaded_pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")

    if "question_bank" not in st.session_state:
        st.session_state.question_bank = {}

    def extract_text_from_pdf(file):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    if uploaded_pdf and st.button("Generate Questions"):
        syllabus_text = extract_text_from_pdf(uploaded_pdf)

        prompt = f"""
        You are a strict JSON generator.

        Output ONLY valid JSON.

        Format:
        {{
            "Unit 1": {{
                "3M": ["Q1", "Q2", "Q3"],
                "5M": ["Q1", "Q2", "Q3"],
                "10M": ["Q1", "Q2", "Q3"]
            }}
        }}

        Syllabus:
        {syllabus_text}
        """

        response = st.session_state.llm.invoke(prompt)

        raw = response.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            try:
                st.session_state.question_bank = json.loads(match.group())
            except:
                st.error("Parsing failed")
        else:
            st.error("Invalid response")

    filter_type = st.selectbox("Filter", ["All", "3M", "5M", "10M"])

    for unit, data in st.session_state.question_bank.items():
        with st.expander(unit, expanded=True):
            for mark_type, questions in data.items():
                if filter_type != "All" and filter_type != mark_type:
                    continue

                st.subheader(mark_type)

                for i, q in enumerate(questions):
                    col1, col2 = st.columns([8, 1])

                    with col1:
                        new_q = st.text_area(f"{unit}_{mark_type}_{i}", value=q)
                        st.session_state.question_bank[unit][mark_type][i] = new_q

                    with col2:
                        if st.button("❌", key=f"del_{unit}_{mark_type}_{i}"):
                            questions.pop(i)
                            st.rerun()

                if st.button(f"➕ Add {mark_type}", key=f"add_{unit}_{mark_type}"):
                    questions.append("New Question")
                    st.rerun()

    if st.button("📄 Export Questions PDF"):
        text = ""
        for unit, data in st.session_state.question_bank.items():
            text += f"# {unit}\n"
            for mtype, qs in data.items():
                text += f"## {mtype}\n"
                for q in qs:
                    text += f"- {q}\n"

        create_pdf(text, "questions.pdf")

        with open("questions.pdf", "rb") as f:
            st.download_button("Download Questions PDF", f, "questions.pdf")
