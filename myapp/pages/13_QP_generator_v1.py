import streamlit as st
import json
import re
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import markdown
import weasyprint
from docx import Document

st.set_page_config(page_title="Question Generator", layout="wide")
st.title("📘 Question Generator")

# ---------------- SESSION ---------------- #
if "question_bank" not in st.session_state:
    st.session_state.question_bank = {}

if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = {"3M": [], "5M": [], "10M": []}

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------- PDF ---------------- #
def create_pdf(text, filename):
    html = markdown.markdown(text)
    weasyprint.HTML(string=html).write_pdf(filename)

# ---------------- WORD ---------------- #
def create_word_doc(text, filename):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(filename)

# ---------------- TEXT EXTRACTION ---------------- #
def extract_text(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

# ---------------- SAFE JSON ---------------- #
#def safe_json_parse(text):
#    try:
#        match = re.search(r"\{.*\}", text, re.DOTALL)
#        parsed = json.loads(match.group())
#
#        if isinstance(parsed, list):
#            parsed = {f"Unit {i+1}": val for i, val in enumerate(parsed)}
#
#
#        return parsed if isinstance(parsed, dict) else {}
#
#    except:
#        return {}

def safe_json_parse(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group())

        # If top-level is list → convert to units
        if isinstance(parsed, list):
            parsed = {f"Unit {i+1}": val for i, val in enumerate(parsed)}

        # 🔥 Normalize structure
        normalized = {}

        for unit, value in parsed.items():

            # Case 1: Already correct dict
            if isinstance(value, dict):
                normalized[unit] = {
                    "3M": value.get("3M", []),
                    "5M": value.get("5M", []),
                    "10M": value.get("10M", [])
                }

            # Case 2: List → assume all are 3M
            elif isinstance(value, list):
                normalized[unit] = {
                    "3M": value,
                    "5M": [],
                    "10M": []
                }

            # Case 3: Unexpected → skip
            else:
                normalized[unit] = {
                    "3M": [],
                    "5M": [],
                    "10M": []
                }

        return normalized

    except Exception as e:
        return {}

def extract_subject_details(text):
    subject_name = ""
    subject_code = ""

    # Common patterns
    code_patterns = [
        r"Subject\s*Code[:\s]*([A-Za-z0-9\-]+)",
        r"Code[:\s]*([A-Za-z0-9\-]+)"
    ]

    name_patterns = [
        r"Subject\s*Name[:\s]*(.+)",
        r"Course\s*Title[:\s]*(.+)",
        r"Title[:\s]*(.+)"
    ]

    # Extract code
    for pattern in code_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            subject_code = match.group(1).strip()
            break

    # Extract name
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            subject_name = match.group(1).strip()
            break

    return subject_name, subject_code

# ---------------- VALIDATION ---------------- #
def validate_selection(selected, req_3m, req_5m, req_10m):
    errors = []

    if len(selected["3M"]) != req_3m:
        errors.append(f"Section A must have {req_3m}, selected {len(selected['3M'])}")

    if len(selected["5M"]) != req_5m:
        errors.append(f"Section B must have {req_5m}, selected {len(selected['5M'])}")

    if len(selected["10M"]) != req_10m:
        errors.append(f"Section C must have {req_10m}, selected {len(selected['10M'])}")

    return errors

# ---------------- UPLOAD ---------------- #
uploaded_pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")

# ---------------- GENERATE ---------------- #
if uploaded_pdf and st.button("Generate Questions"):

    syllabus = extract_text(uploaded_pdf)
    auto_name, auto_code = extract_subject_details(syllabus)

    # Store in session if not already set
    if "auto_subject_name" not in st.session_state:
        st.session_state.auto_subject_name = auto_name

    if "auto_subject_code" not in st.session_state:
        st.session_state.auto_subject_code = auto_code

#    prompt = f"""
#    Extract units and generate questions.
#
#    For EACH unit:
#    - Section A: 10–15 questions
#    - Section B: 6–8 questions
#    - Section C: 3–5 questions
#
#    Return STRICT JSON.
#
#    {syllabus[:4000]}
#    """
    prompt = f"""
    Extract units and generate questions.

    Return STRICT JSON in this format:

    {{
      "Unit 1": {{
        "3M": ["q1", "q2"],
        "5M": ["q1"],
        "10M": ["q1"]
      }},
      "Unit 2": {{
        "3M": [],
        "5M": [],
        "10M": []
      }}
    }}

    Rules:
    - DO NOT return anything outside JSON
    - Each unit must contain keys: 3M, 5M, 10M
    - 3M → 10–15 questions
    - 5M → 6–8 questions
    - 10M → 3–5 questions

    Syllabus:
    {syllabus[:4000]}
    """

    status = st.empty()
    success = False

    for i in range(3):
        status.info(f"🔄 Attempt {i+1}/3")

        try:
            res = st.session_state.llm.invoke(prompt)
            parsed = safe_json_parse(res.content)

            if parsed:
                st.session_state.question_bank = parsed
                success = True
                break

        except:
            continue

    status.empty()

    if success:
        st.success("✅ Questions Generated")
    else:
        st.error("❌ Failed after 3 attempts")

# ---------------- STOP ---------------- #
if not st.session_state.question_bank:
    st.stop()

# ---------------- SUBJECT ---------------- #
st.subheader("💬 Subject Details")

col1, col2, col3 = st.columns(3)
#subject_name = col1.text_input("Subject Name")
#subject_code = col2.text_input("Subject Code")
subject_name = col1.text_input(
    "Subject Name",
    value=st.session_state.get("auto_subject_name", "")
)

subject_code = col2.text_input(
    "Subject Code",
    value=st.session_state.get("auto_subject_code", "")
)
exam_title = col3.text_input("Exam Title", "Mid Semester Exam")

# ---------------- COUNTS ---------------- #
st.subheader("🎯 Question Count")

col1, col2, col3 = st.columns(3)

req_3m = col1.number_input("Section A", min_value=1, value=5)
req_5m = col2.number_input("Section B", min_value=1, value=3)
req_10m = col3.number_input("Section C", min_value=1, value=2)

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
| Section | No. Q | Marks | Total |
|--------|------|------|------|
| A | {req_3m} | {marks_3} | {total_3} |
| B | {req_5m} | {marks_5} | {total_5} |
| C | {req_10m} | {marks_10} | {total_10} |
| **Total** | - | - | **{net_total}** |
""")

# ---------------- SELECTION ---------------- #
units = list(st.session_state.question_bank.keys())
selected_unit = st.selectbox("Select Unit", ["All Units"] + units)
selected_units = units if selected_unit == "All Units" else [selected_unit]

tabs = st.tabs(["Section A", "Section B", "Section C"])

section_map = {"3M": 0, "5M": 1, "10M": 2}
limit_map = {"3M": req_3m, "5M": req_5m, "10M": req_10m}
name_map = {"3M": "Section A", "5M": "Section B", "10M": "Section C"}

for key, idx in section_map.items():

    with tabs[idx]:
        
        selected_list = st.session_state.selected_questions[key]
        st.caption(f"Selected {len(selected_list)} / {limit_map[key]}")

        for unit in selected_units:

#            questions = st.session_state.question_bank.get(unit, {}).get(key, [])
            unit_data = st.session_state.question_bank.get(unit, {})

            if isinstance(unit_data, dict):
                questions = unit_data.get(key, [])
            else:
                questions = []
            st.markdown(f"### {unit}")

            for i, q in enumerate(questions):

                checkbox_key = f"{unit}_{key}_{i}"
                # checked = st.checkbox(q, key=checkbox_key)
                selected_list = st.session_state.selected_questions[key]

                # Disable if limit reached AND this question is not already selected
                disable_flag = (
                    len(selected_list) >= limit_map[key] and q not in selected_list
                )

                checked = st.checkbox(
                    q,
                    key=checkbox_key,
                    disabled=disable_flag
                )

                selected_list = st.session_state.selected_questions[key]

                if checked and q not in selected_list:
                    selected_list.append(q)

                    if len(selected_list) > limit_map[key]:
                        st.toast(
                            f"⚠️ {name_map[key]} exceeded ({len(selected_list)}/{limit_map[key]})",
                            icon="⚠️"
                        )

                elif not checked and q in selected_list:
                    selected_list.remove(q)

# ---------------- LIVE TRACKING ---------------- #
st.subheader("📊 Live Tracking")

tabs = st.tabs(["Section A", "Section B", "Section C"])

for key, idx in section_map.items():

    with tabs[idx]:

        selected = len(st.session_state.selected_questions[key])
        limit = limit_map[key]

        progress = selected / limit if limit > 0 else 0
        progress = min(progress, 1.0)

        st.metric("Selected", selected)
        st.metric("Required", limit)
        st.progress(progress)

        if limit == 0:
            st.info("ℹ️ Set required count")
        elif selected == limit:
            st.success("✅ Completed")
        elif selected < limit:
            st.warning("⚠️ In Progress")
        else:
            st.error(f"❌ Exceeded ({selected}/{limit})")

# ---------------- EXPORT ---------------- #
if st.button("📄 Export File"):

    selected = st.session_state.selected_questions
    errors = validate_selection(selected, req_3m, req_5m, req_10m)

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    text = f"# {exam_title}\n\nSubject: {subject_name}\nCode: {subject_code}\n\n"

    def format_section(qs):
        return "\n".join([f"{i+1}. {q}" for i, q in enumerate(qs)])

    text += "\n\nSection A\n" + format_section(selected["3M"])
    text += "\n\nSection B\n" + format_section(selected["5M"])
    text += "\n\nSection C\n" + format_section(selected["10M"])

    pdf_file = "questions.pdf"
    docx_file = "questions.docx"

    create_pdf(text, pdf_file)
    create_word_doc(text, docx_file)

    col1, col2 = st.columns(2)

    with col1:
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ PDF", f, pdf_file)

    with col2:
        with open(docx_file, "rb") as f:
            st.download_button("⬇️ Word", f, docx_file)

    st.success("✅ Export successful")
