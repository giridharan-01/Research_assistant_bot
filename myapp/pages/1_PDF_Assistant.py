import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

st.title("📄 PDF Assistant")

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")
if "messages" not in st.session_state:
    st.session_state.messages = []


def save_uploaded_file(upload_file):
    os.makedirs("tempDir", exist_ok=True)
    with open(os.path.join("tempDir", upload_file.name), "wb") as f:
        f.write(upload_file.getbuffer())

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded")

if st.button("🚀 Start") and uploaded_files:
    st.toast("Processing PDFs...", icon="⏳")

    docs = []
    for file in uploaded_files:
        save_uploaded_file(file)
        loader = PyPDFLoader(os.path.join("tempDir", file.name))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs.extend(splitter.split_documents(pages))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(docs, embeddings)
    st.session_state.retriever = db.as_retriever()

    st.toast("Done!", icon="✅")

# Chat UI
if "retriever" in st.session_state:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask your question")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        docs = st.session_state.retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        response = st.session_state.llm.invoke(
            f"Answer using context:\n{context}\n\nQuestion:{query}"
        )

        st.chat_message("assistant").write(response.content)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
