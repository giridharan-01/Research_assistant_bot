# =========================
# IMPORTS
# =========================
import streamlit as st
import os
import shutil
import markdown
import weasyprint
import json
import re

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="PDF Assistant", layout="wide")

st.title("Hi, I am Ray..")
st.markdown("Your **PDF Assistant**")

# =========================
# SESSION
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_bank" not in st.session_state:
    st.session_state.question_bank = {}

if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = {"3M": [], "5M": [], "10M": []}

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")

# =========================
# UTILS
# =========================
def save_uploaded_file(upload_file):
    os.makedirs("tempDir", exist_ok=True)
    with open(os.path.join("tempDir", upload_file.name), "wb") as f:
        f.write(upload_file.getbuffer())

def create_pdf(text, filename):
    html = markdown.markdown(text)
    weasyprint.HTML(string=html).write_pdf(filename)

# =========================
# TABS
# =========================
t1, t2 = st.tabs(["📄 PDF Assistant", "📘 Question Generator"])

# =========================
# TAB 1: FIXED CHAT UI
# =========================
with t1:

    st.markdown("""
    <style>
    .chat-box {
        height: 70vh;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 10px;
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

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.from_documents(docs, embeddings)
        st.session_state.retriever = db.as_retriever()

    if "retriever" in st.session_state:

        st.markdown('<div class="chat-box">', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.markdown('</div>', unsafe_allow_html=True)

        query = st.chat_input("Ask your question")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            docs = st.session_state.retriever.get_relevant_documents(query)
            context = "\n".join([d.page_content for d in docs])

            response = st.session_state.llm.invoke(
                f"Answer using context:\n{context}\n\nQuestion:{query}"
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )

            st.rerun()

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("📥 Download Chat"):
            text = ""
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                text += f"**{role}:** {msg['content']}\n\n"

            create_pdf(text, "chat.pdf")

            with open("chat.pdf", "rb") as f:
                st.download_button("Download", f, "chat.pdf")

# =========================
# TAB 2: QUESTION SYSTEM
# =========================
with t2:

    uploaded_pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")

    def extract_text(file):
        reader = PdfReader(file)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

    # =========================
    # GENERATE QUESTIONS
    # =========================
    if uploaded_pdf and st.button("Generate Questions"):

        syllabus = extract_text(uploaded_pdf)

        prompt = f"""
        You are an expert university question paper setter.

        Generate a LARGE number of HIGH-QUALITY exam questions.

        STRICT RULES:
        - DO NOT use placeholders like q1, q2
        - Questions must be complete and meaningful
        - Use Bloom’s taxonomy verbs
        - No repetition
        - Cover all topics

        REQUIREMENTS:
        - 3M: At least 20 questions
        - 5M: At least 12 questions
        - 10M: At least 8 questions

        RETURN ONLY VALID JSON

        FORMAT:
        {{
          "Unit 1": {{
            "3M": ["Define...", "..."],
            "5M": ["Explain...", "..."],
            "10M": ["Analyze...", "..."]
          }},
          "Unit 2": {{
            "3M": [...],
            "5M": [...],
            "10M": [...]
          }}
        }}

        Syllabus:
        {syllabus}
        """

        res = st.session_state.llm.invoke(prompt)

        match = re.search(r"\{.*\}", res.content, re.DOTALL)

        if match:
            try:
                parsed = json.loads(match.group())

                # Validate structure
                for unit, val in parsed.items():
                    if not isinstance(val, dict):
                        continue
                    for k in ["3M", "5M", "10M"]:
                        if k not in val or not isinstance(val[k], list):
                            val[k] = []

                st.session_state.question_bank = parsed

            except:
                st.error("⚠️ JSON parsing failed")

    # =========================
    # UNIT SELECTBOX
    # =========================
    st.subheader("🎯 Select Unit")

    all_units = list(st.session_state.question_bank.keys())

    unit_option = st.selectbox(
        "Choose Unit",
        ["All Units"] + all_units if all_units else ["All Units"]
    )

    if unit_option == "All Units":
        selected_unit_names = all_units
    else:
        selected_unit_names = [unit_option]

    # =========================
    # LIMITS
    # =========================
    limits = {"3M": 10, "5M": 5, "10M": 4}

    # =========================
    # DISPLAY QUESTIONS
    # =========================
    for mark_type in ["3M", "5M", "10M"]:

        st.subheader(f"{mark_type} Questions")

        selected = st.session_state.selected_questions[mark_type]

        for unit in selected_unit_names:

            if unit not in st.session_state.question_bank:
                continue

            data = st.session_state.question_bank[unit]

            if not isinstance(data, dict):
                continue

            questions = data.get(mark_type, [])

            if not questions:
                st.warning(f"No {mark_type} questions in {unit}")
                continue

            st.markdown(f"### 📘 {unit}")

            for i, q in enumerate(questions):

                key = f"{unit}_{mark_type}_{i}"
                checked = st.checkbox(q, key=key)

                if checked and q not in selected:
                    if len(selected) < limits[mark_type]:
                        selected.append(q)
                    else:
                        st.warning(f"⚠️ Max {limits[mark_type]} reached")

                elif not checked and q in selected:
                    selected.remove(q)

    # =========================
    # EXPORT
    # =========================
    st.divider()

    if st.button("📄 Export Selected Questions"):

        text = ""

        for mtype, qs in st.session_state.selected_questions.items():
            text += f"## {mtype}\n"
            for q in qs:
                text += f"- {q}\n"

        create_pdf(text, "questions.pdf")

        with open("questions.pdf", "rb") as f:
            st.download_button("Download PDF", f, "questions.pdf")
