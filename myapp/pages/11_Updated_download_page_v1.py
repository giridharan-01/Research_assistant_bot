import streamlit as st
import markdown
import weasyprint
from datetime import datetime

st.set_page_config(page_title="Download Chat", layout="wide")
st.title("⬇️ Download Chat")

# ---------------- CHECK ---------------- #
if "messages" not in st.session_state or not st.session_state.messages:
    st.error("❌ No chat available. Please interact with the assistant first.")
    st.stop()

messages = st.session_state.messages

# ---------------- GROUP INTO QA ---------------- #
qa_pairs = []
temp_q = None

for msg in messages:
    if msg["role"] == "user":
        temp_q = msg["content"]
    elif msg["role"] == "assistant" and temp_q:
        qa_pairs.append((temp_q, msg["content"]))
        temp_q = None

if not qa_pairs:
    st.error("❌ No valid Q&A pairs found.")
    st.stop()

# ---------------- SELECT ALL / DESELECT ALL ---------------- #
st.subheader("☑️ Select Chats to Export")

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Select All"):
        for i in range(len(qa_pairs)):
            st.session_state[f"chk_{i}"] = True

with col2:
    if st.button("❌ Deselect All"):
        for i in range(len(qa_pairs)):
            st.session_state[f"chk_{i}"] = False

st.markdown("---")

# ---------------- CHECKBOX LIST ---------------- #
selected_indices = []

for i, (q, a) in enumerate(qa_pairs):
    default_val = st.session_state.get(f"chk_{i}", True)

    checked = st.checkbox(
        f"Q{i+1}: {q[:80]}...",
        value=default_val,
        key=f"chk_{i}"
    )

    with st.expander(f"View Q{i+1}"):
        st.markdown(f"**Q{i+1}. {q}**")
        st.markdown(f"**Answer:** {a}")

    if checked:
        selected_indices.append(i)

# ---------------- CUSTOM TITLE ---------------- #
st.subheader("📝 Customize PDF")

chat_title = st.text_input("Enter Title", "PDF Chat Export")

# ---------------- FORMAT FUNCTION ---------------- #
def format_chat(qa_pairs, selected_indices, title):

    timestamp = datetime.now().strftime("%d %B %Y, %I:%M %p")

    text = f"# {title}\n\n"
    text += f"📅 Generated on: {timestamp}\n\n---\n\n"

    q_no = 1

    for idx in selected_indices:
        q, a = qa_pairs[idx]

        text += f"## Q{q_no}. {q}\n\n"
        text += f"**Answer:**\n{a}\n\n---\n\n"

        q_no += 1

    return text

# ---------------- PREVIEW ---------------- #
st.subheader("👀 Preview Selected Content")

if selected_indices:
    formatted_text = format_chat(qa_pairs, selected_indices, chat_title)
    st.markdown(formatted_text)
else:
    st.warning("⚠️ No questions selected. Please select at least one.")

# ---------------- PDF FUNCTION ---------------- #
def create_pdf(text, filename):
    html_content = markdown.markdown(text)

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial;
                padding: 20px;
            }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    weasyprint.HTML(string=html).write_pdf(filename)

# ---------------- CENTERED DOWNLOAD BUTTON ---------------- #
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("📥 Generate & Download PDF", use_container_width=True):

        if not selected_indices:
            st.error("❌ Please select at least one question before generating PDF.")
        else:
#            create_pdf(formatted_text, "chat.pdf")
#
#            with open("chat.pdf", "rb") as f:
#                st.download_button(
#                    "⬇️ Download PDF",
#                    f,
#                    "chat.pdf",
#                    mime="application/pdf",
#                    use_container_width=True
#                )
            # Clean filename (remove spaces/special chars)
            safe_title = chat_title.strip().replace(" ", "_")

            file_name = f"{safe_title}.pdf"

            create_pdf(formatted_text, file_name)

            with open(file_name, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF",
                    f,
                    file_name,
                    mime="application/pdf",
                    use_container_width=True
                )

            st.success("✅ PDF generated successfully!")
