import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

st.set_page_config(page_title="PDF Assistant", layout="wide")
st.title("📄 PDF Assistant (RAG Powered)")

#st.session_state.db = db
#st.session_state.retriever = db.as_retriever()
#st.session_state.chunks = docs   # 🔥 VERY IMPORTANT
#st.session_state.embeddings = embeddings

# ---------------- SESSION ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------- FILE SAVE ---------------- #
def save_uploaded_file(upload_file):
    os.makedirs("tempDir", exist_ok=True)
    with open(os.path.join("tempDir", upload_file.name), "wb") as f:
        f.write(upload_file.getbuffer())

# ---------------- CHAT HISTORY ---------------- #
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ---------------- CREATE RAG ---------------- #
def create_rag_pipeline(retriever):

    llm = st.session_state.llm

    # Contextual question reformulation
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rephrase the question into a standalone question. Do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using the provided context. "
         "If unknown, say 'I don't know'. Keep it concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    rag_pipeline = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return rag_pipeline

# ---------------- FILE UPLOAD ---------------- #
uploaded_files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded")

# ---------------- PROCESS ---------------- #
if st.button("🚀 Start") and uploaded_files:
    st.toast("Processing PDFs...", icon="⏳")

    docs = []

    for file in uploaded_files:
        save_uploaded_file(file)

        loader = PyPDFLoader(os.path.join("tempDir", file.name))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        docs.extend(splitter.split_documents(pages))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#    db = FAISS.from_documents(docs, embeddings)
#    retriever = db.as_retriever()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # ✅ STORE FOR REUSE (VERY IMPORTANT)
    st.session_state.db = db
    st.session_state.retriever = retriever
    st.session_state.chunks = docs
    st.session_state.embeddings = embeddings

    # ✅ CREATE RAG PIPELINE
    st.session_state.rag = create_rag_pipeline(retriever)

    st.toast("Done!", icon="✅")

# ---------------- CHAT UI ---------------- #
if "rag" in st.session_state:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask your question")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        response = st.session_state.rag.invoke(
            {"input": query},
            config={"configurable": {"session_id": "user123"}}
        )["answer"]

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
