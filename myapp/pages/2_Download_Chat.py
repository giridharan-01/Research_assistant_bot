import streamlit as st
import markdown
import weasyprint

st.title("⬇️ Download Chat")

def create_pdf(text, filename):
    html = markdown.markdown(text)
    weasyprint.HTML(string=html).write_pdf(filename)

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.toast("Chat cleared!", icon="🗑️")

with col2:
    if st.button("📥 Download Chat"):

        text = ""
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                text += f"**Q{int((i/2)+1)}. {msg['content']}**\n\n"
            else:
                text += f"**Answer:** {msg['content']}\n\n"

        create_pdf(text, "chat.pdf")

        with open("chat.pdf", "rb") as f:
            st.download_button("Download PDF", f, "chat.pdf")
