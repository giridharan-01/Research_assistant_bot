import streamlit as st
import markdown
import weasyprint
from langchain_openai import ChatOpenAI

st.title("🤖 ChatGPT Assistant")

# =========================
# INIT SESSION
# =========================
if "chatgpt_messages" not in st.session_state:
    st.session_state.chatgpt_messages = []

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")

# =========================
# PDF CREATION
# =========================
def create_pdf(text, filename):
    html = markdown.markdown(text)
    weasyprint.HTML(string=html).write_pdf(filename)

# =========================
# CHAT DISPLAY
# =========================
for msg in st.session_state.chatgpt_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# USER INPUT
# =========================
query = st.chat_input("Ask anything...")

if query:
    st.session_state.chatgpt_messages.append({"role": "user", "content": query})

    st.toast("🤖 Thinking...", icon="💡")

    st.chat_message("user").write(query)

    response = st.session_state.llm.invoke(query)

    st.chat_message("assistant").write(response.content)

    st.session_state.chatgpt_messages.append(
        {"role": "assistant", "content": response.content}
    )

# =========================
# ACTION BUTTONS
# =========================
st.divider()

col1, col2 = st.columns(2)

# CLEAR CHAT
with col1:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chatgpt_messages = []
        st.toast("Chat cleared!", icon="🗑️")

# DOWNLOAD CHAT
with col2:
    if st.button("📥 Download Chat", use_container_width=True):

        st.toast("📄 Generating PDF...", icon="⏳")

        text = ""
        for i, msg in enumerate(st.session_state.chatgpt_messages):
            if msg["role"] == "user":
                text += f"**Q{int((i/2)+1)}. {msg['content']}**\n\n"
            else:
                text += f"**Answer:** {msg['content']}\n\n"

        create_pdf(text, "chatgpt_chat.pdf")

        st.toast("✅ Ready!", icon="📥")

        with open("chatgpt_chat.pdf", "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                "chatgpt_chat.pdf",
                use_container_width=True
            )
