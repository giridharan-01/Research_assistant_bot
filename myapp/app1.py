import streamlit as st

st.set_page_config(page_title="JIM - PDF Assistant", layout="wide")

st.title("🤖 Welcome to JIM")
st.markdown("### Your **PDF Assistant - Just In Mins**")

st.info("👉 Use the sidebar to navigate between features")

# Initialize shared state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_bank" not in st.session_state:
    st.session_state.question_bank = {}

if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = {"3M": [], "5M": [], "10M": []}
