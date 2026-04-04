import streamlit as st
import json
import re
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import markdown
import weasyprint

st.set_page_config(page_title="Question Generator", layout="wide")
st.title("📘 Question Generator")

# ---------------- SESSION ---------------- #
if "question_bank" not in st.session_state:
    st.session_state.question_bank = {}

if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = {"3M": [], "5M": [], "10M": []}

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------- PDF ---------------- #
def create_pdf(text, filename):
    html = markdown.markdown(text)
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial; padding: 20px; }}
            h1 {{ text-align: center; }}
        </style>
    </head>
    <body>{html}</body>
    </html>
    """
    weasyprint.HTML(string=styled_html).write_pdf(filename)

# ---------------- EXTRACT TEXT ---------------- #
def extract_text(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

# ---------------- VALIDATION ---------------- #
def validate_counts(data):
    for unit, sections in data.items():
        if len(sections.get("3M", [])) < 20:
            return False
        if len(sections.get("5M", [])) < 10:
            return False
        if len(sections.get("10M", [])) < 5:
            return False
    return True

def validate_selection(selected, req_3m, req_5m, req_10m):

    errors = []

    if len(selected["3M"]) != req_3m:
        errors.append(f"Section A must have {req_3m} questions (Selected: {len(selected['3M'])})")

    if len(selected["5M"]) != req_5m:
        errors.append(f"Section B must have {req_5m} questions (Selected: {len(selected['5M'])})")

    if len(selected["10M"]) != req_10m:
        errors.append(f"Section C must have {req_10m} questions (Selected: {len(selected['10M'])})")

    return errors

# ---------------- SUBJECT ---------------- #
def extract_subject_details(syllabus):

    prompt = f"""
    Extract subject name and subject code in JSON:
    {{
        "subject_name": "...",
        "subject_code": "..."
    }}
    {syllabus}
    """

    for _ in range(3):
        try:
            res = st.session_state.llm.invoke(prompt)
            match = re.search(r"\{.*\}", res.content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return parsed.get("subject_name", ""), parsed.get("subject_code", "")
        except:
            continue

    return "", ""

# ---------------- UPLOAD ---------------- #
uploaded_pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")

# ---------------- GENERATE ---------------- #
if uploaded_pdf and st.button("Generate Questions"):

    syllabus = extract_text(uploaded_pdf)

    s_name, s_code = extract_subject_details(syllabus)
    st.session_state.subject_name = s_name
    st.session_state.subject_code = s_code

    prompt = f"""
    Generate questions in JSON.

    Each unit must contain:
    3M: 20
    5M: 10
    10M: 5

    {syllabus}
    """

    for i in range(3):
        try:
            res = st.session_state.llm.invoke(prompt)
            match = re.search(r"\{.*\}", res.content, re.DOTALL)
            parsed = json.loads(match.group())

            if not validate_counts(parsed):
                raise ValueError("Incomplete generation")

            st.session_state.question_bank = parsed
            st.success("✅ Questions Generated")
            break

        except:
            st.warning("Retrying...")

# ---------------- STOP ---------------- #
if not st.session_state.question_bank:
    st.stop()

# ---------------- SUBJECT UI ---------------- #
st.subheader("💬 Subject Details")

col1, col2, col3 = st.columns(3)

subject_name = col1.text_input("Subject Name", st.session_state.get("subject_name", ""))
subject_code = col2.text_input("Subject Code", st.session_state.get("subject_code", ""))
exam_title = col3.text_input("Exam Title", "Mid Semester Exam")

# ---------------- COUNTS ---------------- #
st.subheader("🎯 Question Count")

col1, col2, col3 = st.columns(3)

req_3m = col1.number_input("Section A", min_value=0)
req_5m = col2.number_input("Section B", min_value=0)
req_10m = col3.number_input("Section C", min_value=0)

# ---------------- MARKS TABLE ---------------- #
st.subheader("📊 Marks Distribution")

col1, col2, col3 = st.columns(3)

marks_3 = col1.number_input("Marks per Question (A)", value=3)
marks_5 = col2.number_input("Marks per Question (B)", value=5)
marks_10 = col3.number_input("Marks per Question (C)", value=10)

total_3 = req_3m * marks_3
total_5 = req_5m * marks_5
total_10 = req_10m * marks_10
net_total = total_3 + total_5 + total_10

st.markdown(f"""
| Section | No. Questions | Marks/Q | Total |
|--------|--------------|--------|------|
| A | {req_3m} | {marks_3} | {total_3} |
| B | {req_5m} | {marks_5} | {total_5} |
| C | {req_10m} | {marks_10} | {total_10} |
| **Total** | - | - | **{net_total}** |
""")

# ---------------- UNITS ---------------- #
units = list(st.session_state.question_bank.keys())

selected_unit = st.selectbox("Select Unit", ["All Units"] + units)
selected_units = units if selected_unit == "All Units" else [selected_unit]

# ---------------- TABS ---------------- #
tab1, tab2, tab3 = st.tabs(["Section A", "Section B", "Section C"])

tabs_map = {"3M": tab1, "5M": tab2, "10M": tab3}
section_names = {"3M": "Section A", "5M": "Section B", "10M": "Section C"}

limit_map = {"3M": req_3m, "5M": req_5m, "10M": req_10m}

for mark_type, tab in tabs_map.items():

    with tab:

        selected_q = st.session_state.selected_questions[mark_type]

        for unit in selected_units:

            questions = st.session_state.question_bank[unit].get(mark_type, [])

            st.markdown(f"### {unit}")

            for i, q in enumerate(questions):

                key = f"{unit}_{mark_type}_{i}"
                checked = st.checkbox(q, key=key)

                if checked and q not in selected_q:
                    if len(selected_q) < limit_map[mark_type]:
                        selected_q.append(q)
                    else:
                        st.warning(f"⚠️ Max {limit_map[mark_type]} allowed")
                        st.session_state[key] = False

                elif not checked and q in selected_q:
                    selected_q.remove(q)

# ---------------- STATUS ---------------- #
st.subheader("📊 Selection Status")

col1, col2, col3 = st.columns(3)

col1.metric("Section A", f"{len(st.session_state.selected_questions['3M'])}/{req_3m}")
col2.metric("Section B", f"{len(st.session_state.selected_questions['5M'])}/{req_5m}")
col3.metric("Section C", f"{len(st.session_state.selected_questions['10M'])}/{req_10m}")

# ---------------- EXPORT ---------------- #
if st.button("📄 Export PDF"):

    selected = st.session_state.selected_questions

    errors = validate_selection(selected, req_3m, req_5m, req_10m)

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    text = f"# {exam_title}\n\n"
    text += f"**Subject:** {subject_name}\n\n"
    text += f"**Code:** {subject_code}\n\n---\n\n"

    text += "## Marks Distribution\n\n"
    text += f"""
| Section | No. Questions | Marks/Q | Total |
|--------|--------------|--------|------|
| A | {req_3m} | {marks_3} | {total_3} |
| B | {req_5m} | {marks_5} | {total_5} |
| C | {req_10m} | {marks_10} | {total_10} |
| **Total** | - | - | **{net_total}** |
"""

    def add_section(title, questions):
        s = f"\n## {title}\n\n"
        for i, q in enumerate(questions, 1):
            s += f"{i}. {q}\n\n"
        return s

    text += add_section("Section A", selected["3M"])
    text += add_section("Section B", selected["5M"])
    text += add_section("Section C", selected["10M"])

    file_name = f"{subject_code}_{exam_title.replace(' ', '_')}.pdf"

    create_pdf(text, file_name)

    with open(file_name, "rb") as f:
        st.download_button("⬇️ Download", f, file_name)

    st.success("✅ PDF Generated")
