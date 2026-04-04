import streamlit as st
import json
import re
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import markdown
import weasyprint
from docx import Document

# ---------------- PAGE CONFIG ---------------- #
# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Question Generator", layout="wide")
st.title("📘 Question Generator")

# ---------------- SESSION ---------------- #
if "question_bank" not in st.session_state:
    st.session_state.question_bank = {}

if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = {"3M": [], "5M": [], "10M": []}

# ✅ LLM initialized once
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------- PDF ---------------- #
def create_pdf(text, filename):
    html_content = markdown.markdown(text)

    # ✅ Add styling (VERY IMPORTANT)
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                padding: 20px;
            }}
            h1, h2, h3 {{
                text-align: center;
                margin-bottom: 10px;
            }}
            p {{
                margin: 5px 0;
            }}
            ol {{
                margin-left: 20px;
            }}
            li {{
                margin-bottom: 6px;
            }}
            strong {{
                font-size: 16px;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    weasyprint.HTML(string=styled_html).write_pdf(filename)

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

# ---------------- SAFE JSON PARSE ---------------- #
# ✅ UPDATED: Now handles subject + units + normalization
def safe_json_parse(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group())

        # ✅ Extract metadata
        subject_name = parsed.get("subject_name", "")
        subject_code = parsed.get("subject_code", "")
        units = parsed.get("units", {})

        normalized = {}

        # ✅ Normalize question structure
        for unit, value in units.items():

            if isinstance(value, dict):
                normalized[unit] = {
                    "3M": value.get("3M", []),
                    "5M": value.get("5M", []),
                    "10M": value.get("10M", [])
                }
            else:
                normalized[unit] = {
                    "3M": [],
                    "5M": [],
                    "10M": []
                }

        return {
            "subject_name": subject_name,
            "subject_code": subject_code,
            "question_bank": normalized
        }

    except:
        return {}

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

    # ✅ UPDATED PROMPT: Includes subject extraction
    prompt = f"""
    You are an expert university question paper setter.

    TASK:
    1. Identify all units from the syllabus
    2. Extract ALL topics under each unit
    3. Ensure EVERY topic is covered
    4. Generate COMPLETE exam questions (NOT topics)

    STRICT QUESTION COUNT (MANDATORY):
    For EACH UNIT:
    - Section A (3M): EXACTLY 12 to 15 questions
    - Section B (5M): EXACTLY 6 to 8 questions
    - Section C (10M): EXACTLY 5 to 6 questions

    ❗ DO NOT generate fewer questions
    ❗ If unsure, generate MORE within the range

    QUESTION STYLE RULES:

    Section A (3M):
    - Use WH questions (What, Why, When, Where)
    - Use: Define, List, State, Identify

    Section B (5M):
    - Use: Explain, Compare, Differentiate, Illustrate

    Section C (10M):
    - Use: Analyse, Evaluate, Discuss, Justify
    - MUST include 1–2 CASE STUDY questions for IMPORTANT units

    IMPORTANT UNIT RULE:
    - If a unit has more topics → treat it as IMPORTANT
    - Add case study questions ONLY for important units

    CASE STUDY FORMAT:
    - Provide a short scenario/problem
    - Then ask analytical questions

    STRICT RULES:
    - Questions MUST be full sentences
    - Questions MUST NOT be keywords
    - NO repetition
    - COVER ALL topics

    OUTPUT FORMAT (STRICT JSON):

    {{
      "subject_name": "string",
      "subject_code": "string",
      "units": {{
        "Unit 1": {{
          "topics": ["topic1", "topic2"],
          "3M": [],
          "5M": [],
          "10M": []
        }}
      }}
    }}

    VALIDATION BEFORE OUTPUT:
    - Ensure Section A has at least 12 questions
    - Ensure Section B has at least 6 questions
    - Ensure Section C has at least 5 questions

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
                # ✅ Store everything in session
                st.session_state.question_bank = parsed["question_bank"]
                st.session_state.auto_subject_name = parsed["subject_name"]
                st.session_state.auto_subject_code = parsed["subject_code"]

                success = True
                break

        except:
            continue

    status.empty()

    if success:
        st.success("✅ Questions Generated")

        # ✅ Fallback warning
        if not st.session_state.auto_subject_name:
            st.warning("⚠️ Subject name not detected. Please enter manually.")

    else:
        st.error("❌ Failed after 3 attempts")

# ---------------- STOP ---------------- #
if not st.session_state.question_bank:
    st.stop()

# ---------------- SUBJECT ---------------- #
st.subheader("💬 Subject Details")

col1, col2, col3 = st.columns(3)

# ✅ Auto-filled using LLM output
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
        limit = limit_map[key]

        # ✅ GLOBAL LOCK FIX (applies across ALL units)
        section_locked = len(selected_list) >= limit

        # ✅ Live tracker
        st.caption(f"Selected {len(selected_list)} / {limit}")

        # ✅ Toast once when completed
        if section_locked:
            st.toast(f"✅ {name_map[key]} selection completed", icon="✅")

        for unit in selected_units:

            unit_data = st.session_state.question_bank.get(unit, {})
            questions = unit_data.get(key, []) if isinstance(unit_data, dict) else []

            st.markdown(f"### {unit}")

            for i, q in enumerate(questions):

                checkbox_key = f"{unit}_{key}_{i}"

                # ✅ FIX: Disable ALL units when limit reached
                checked = st.checkbox(
                    q,
                    key=checkbox_key,
                    disabled=(section_locked and q not in selected_list)
                )

                if checked and q not in selected_list:
                    selected_list.append(q)
                    st.rerun() 

                elif not checked and q in selected_list:
                    selected_list.remove(q)
                    st.rerun() 

# ---------------- EXPORT ---------------- #

# ✅ STEP 1: Generate files only once
if st.button("📄 Generate Download Files"):

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

    # ✅ FIX: Store files in session (persist across reruns)
    with open(pdf_file, "rb") as f:
        st.session_state.pdf_data = f.read()

    with open(docx_file, "rb") as f:
        st.session_state.docx_data = f.read()

    st.success("✅ Files Ready for Download")


# ✅ STEP 2: Persistent download buttons
if "pdf_data" in st.session_state and "docx_data" in st.session_state:

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇️ Download PDF",
            st.session_state.pdf_data,
            "questions.pdf"
        )

    with col2:
        st.download_button(
            "⬇️ Download Word",
            st.session_state.docx_data,
            "questions.docx"
        )
