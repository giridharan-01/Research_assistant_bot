import streamlit as st
import markdown
import weasyprint
from datetime import datetime

st.set_page_config(page_title="Download Chat", layout="wide")
st.title("⬇️ Download Chat")

# ---------------- SESSION CHECK ---------------- #
if "messages" not in st.session_state or not st.session_state.messages:
    st.warning("⚠️ No chat available. Please interact with the PDF Assistant first.")
    st.stop()

messages = st.session_state.messages

# ---------------- CHAT SUMMARY ---------------- #
st.subheader("📊 Chat Summary")

num_messages = len(messages)
num_questions = sum(1 for m in messages if m["role"] == "user")

col1, col2 = st.columns(2)
col1.metric("💬 Total Messages", num_messages)
col2.metric("❓ Questions Asked", num_questions)

st.markdown("---")

# ---------------- CUSTOM TITLE ---------------- #
st.subheader("📝 Customize Export")

chat_title = st.text_input("Enter Chat Title", value="PDF Assistant Chat Export")

# ---------------- FORMAT FUNCTION ---------------- #
def format_chat(messages, title):

    timestamp = datetime.now().strftime("%d %B %Y, %I:%M %p")

    text = f"# {title}\n\n"
    text += f"📅 Generated on: {timestamp}\n\n"
    text += "---\n\n"

    q_no = 1

    for msg in messages:
        if msg["role"] == "user":
            text += f"## Q{q_no}. {msg['content']}\n\n"
        else:
            text += f"**Answer:**\n{msg['content']}\n\n---\n\n"
            q_no += 1

    return text

# ---------------- PREVIEW ---------------- #
st.subheader("👀 Preview")

formatted_text = format_chat(messages, chat_title)
st.markdown(formatted_text)

# ---------------- PDF GENERATION ---------------- #
def create_pdf(text, filename):
    html_content = markdown.markdown(text)

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
            }}
            h2 {{
                color: #34495e;
            }}
            p {{
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    weasyprint.HTML(string=html).write_pdf(filename)

# ---------------- ACTION BUTTONS ---------------- #
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.toast("Chat cleared!", icon="🗑️")

with col2:
    if st.button("📥 Generate & Download PDF"):

        create_pdf(formatted_text, "chat.pdf")

        with open("chat.pdf", "rb") as f:
            st.download_button(
                label="⬇️ Click to Download",
                data=f,
                file_name="chat.pdf",
                mime="application/pdf"
            )

        st.success("✅ PDF generated successfully!")
