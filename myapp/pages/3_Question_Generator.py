import streamlit as st
import json
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
import markdown
import weasyprint
import json
import re

st.title("📘 Question Generator")

if "question_bank" not in st.session_state:
    st.session_state.question_bank = {}

if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = {"3M": [], "5M": [], "10M": []}

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")

def create_pdf(text, filename):
    html = markdown.markdown(text)
    weasyprint.HTML(string=html).write_pdf(filename)

def extract_text(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

uploaded_pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")


if uploaded_pdf and st.button("Generate Questions"):
    st.toast("📘 Generating questions...", icon="⏳")
    syllabus = extract_text(uploaded_pdf)

    prompt = f"""
    Generate structured university questions in JSON.

    REQUIREMENTS:
    - 3M: 20 questions
    - 5M: 12 questions
    - 10M: 8 questions

    FORMAT:
    {{
      "Unit 1": {{
        "3M": ["..."],
        "5M": ["..."],
        "10M": ["..."]
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
            st.session_state.question_bank = parsed
            st.toast("✅ Questions generated!", icon="🎯")
        except:
            st.error("⚠️ JSON parsing failed")

st.subheader("🎯 Select Unit")

all_units = list(st.session_state.question_bank.keys())

unit_option = st.selectbox(
    "Choose Unit",
    ["All Units"] + all_units if all_units else ["All Units"]
)

selected_unit_names = all_units if unit_option == "All Units" else [unit_option]

limits = {"3M": 10, "5M": 5, "10M": 4}

# =========================
# TABS FOR MARK TYPES
# =========================
tab1, tab2, tab3 = st.tabs(["🟢 3 Marks", "🟡 5 Marks", "🔴 10 Marks"])

tabs_map = {
    "3M": tab1,
    "5M": tab2,
    "10M": tab3
}

for mark_type, tab in tabs_map.items():

    with tab:

        st.subheader(f"{mark_type} Questions")

        selected_q = st.session_state.selected_questions[mark_type]

        for unit in selected_unit_names:

            if unit not in st.session_state.question_bank:
                continue

            questions = st.session_state.question_bank[unit].get(mark_type, [])

            if not questions:
                st.warning(f"No {mark_type} questions in {unit}")
                continue

            st.markdown(f"### 📘 {unit}")

            for i, q in enumerate(questions):

                key = f"{unit}_{mark_type}_{i}"
                checked = st.checkbox(q, key=key)

                if checked and q not in selected_q:
                    if len(selected_q) < limits[mark_type]:
                        selected_q.append(q)
                    else:
                        st.warning(f"⚠️ Max {limits[mark_type]} questions allowed")

                elif not checked and q in selected_q:
                    selected_q.remove(q)
#st.subheader("🎯 Select Unit")
#
#all_units = list(st.session_state.question_bank.keys())
#
#unit_option = st.selectbox(
#    "Choose Unit",
#    ["All Units"] + all_units if all_units else ["All Units"]
#)
#
#selected_unit_names = all_units if unit_option == "All Units" else [unit_option]
#
#limits = {"3M": 10, "5M": 5, "10M": 4}
#
#for mark_type in ["3M", "5M", "10M"]:
#
#    st.subheader(f"{mark_type} Questions")
#
#    selected_q = st.session_state.selected_questions[mark_type]
#
#    for unit in selected_unit_names:
#
#        if unit not in st.session_state.question_bank:
#            continue
#
#        questions = st.session_state.question_bank[unit].get(mark_type, [])
#
#        st.markdown(f"### 📘 {unit}")
#
#        for i, q in enumerate(questions):
#
#            key = f"{unit}_{mark_type}_{i}"
#            checked = st.checkbox(q, key=key)
#
#            if checked and q not in selected_q:
#                if len(selected_q) < limits[mark_type]:
#                    selected_q.append(q)
#                else:
#                    st.warning(f"⚠️ Max {limits[mark_type]} reached")
#
#            elif not checked and q in selected_q:
#                selected_q.remove(q)
#
#st.divider()

if st.button("📄 Export Selected Questions"):
    st.toast("📄 Creating Question paper PDF...", icon="⏳")
    text = ""

    for mtype, qs in st.session_state.selected_questions.items():
        text += f"## {mtype}\n"
        for q in qs:
            text += f"- {q}\n"

    create_pdf(text, "questions.pdf")
    st.toast("✅ PDF ready for download!", icon="📥")
    with open("questions.pdf", "rb") as f:
        st.download_button("Download PDF", f, "questions.pdf")

