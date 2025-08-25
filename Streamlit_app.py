import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os, shutil
from dotenv import load_dotenv
import pdfkit
import datetime
# from langchain import PromptTemplate

load_dotenv()

st.set_page_config(page_title="Chat with PDF", page_icon=":books:", layout="wide")
st.title("PDF Assistant")

store = {}

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

def save_uploaded_file(upload_file):
    with open(os.path.join("tempDir", upload_file.name), "wb") as f:
        f.write(upload_file.getbuffer())

def delete_contents():
    folder = os.getcwd() + "/tempDir"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def extract_metadata(file_path):
    reader = PdfReader(file_path)
    info = reader.metadata
    title = info.title if info.title else "Unknown Title"
    authors = info.author if info.author else "Unknown Authors"
    return title, authors

def initialize_setup(doc_pages):
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv('OPENAI_API_KEY'))
    vectorstore = Chroma.from_documents(documents=doc_pages, 
                                        embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vectorstore.as_retriever()
    return llm, retriever

def create_rag_chain(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

st.cache_data()
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_rag_pipeline(rag_chain):
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
st.cache_data()
def get_response(obj, query):
    response = obj.invoke(
        {"input": query},
        config={"configurable": {"session_id": "user123"}}
    )["answer"]
    return response

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv('OPENAI_API_KEY'))


def summarize_documents(split_docs):
    llm = st.session_state.llm  # Use the already initialized LLM
    # prompt_template = """
    # Please summarize the sentence according to the following REQUEST.
    # REQUEST:
    # 1. Summarize the main points in bullet points.
    # 2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
    # 3. Use various emojis to make the summary more interesting.
    # 4. DO NOT include any unnecessary information.

    # CONTEXT:
    # {context}

    # SUMMARY:

    # """ 
    # prompt = PromptTemplate.from_template(template=prompt_template)
    # chain = load_summarize_chain(llm=llm,prompt=prompt, chain_type="map_reduce")
    # chain = load_summarize_chain(llm=llm, chain_type="refine")
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summary = chain.invoke(split_docs)
    return summary


project_idea_template = (
    """"Based on the degree level {degree_level} and technology {technology}, "
    "generate up to 5 unique project titles in a list. Strictly generate titles only"
    Output : 
    """
)
project_description_template = (
    "You are Researcher and Your task is to generate content with the following fields "
    "For the project titled {project_title}, provide the following details in a List:\n"
    "- Problem Statement\n"
    "- Technology/Tools Used\n"
    "- Use Case/Application\n"
    "- Degree Level : {degree_level} degree\n"
    "- Abstract\n"
    "- Objectives\n"
    "- Conclusion/Future Work\n"
    "- Keywords\n"
    "- Target Audience"
)

project_format_template = (
    """You are a DataAnalyst and Your task is to generate the key-value pairs for the given context 
    For the given {details}, provide the set of key-value pairs in a List:\n
    Lets think step by step
    Output:
    """

)

# openai_api_key = os.getenv('OPENAI_API_KEY')
# Create model with gpt-4o-mini
model = ChatOpenAI(model="gpt-4o-mini")

# Create parsers
# parser = CommaSeparatedListOutputParser()
parser = StrOutputParser()

json_parser = JsonOutputParser()

# Create chains for generating project ideas and detailed descriptions
project_idea_prompt = ChatPromptTemplate.from_messages([
    ('system', project_idea_template)
])
project_description_prompt = ChatPromptTemplate.from_messages([
    ('system', project_description_template)
])

format_prompt = ChatPromptTemplate.from_messages([
    ('system', project_format_template)
])

project_idea_chain = project_idea_prompt | model | parser
project_description_chain = project_description_prompt | model | parser
format_chain = format_prompt | model 
# project_idea_chain = project_idea_prompt | model
# project_description_chain = project_description_prompt | model 

# Function to generate project ideas
st.cache_data()
def generate_project_ideas(degree_level, technology):
    response = project_idea_chain.invoke({"degree_level": degree_level, "technology": technology})
    return response

# Function to generate project details for a specific project title
st.cache_data()
def generate_project_details(project_title,degree_level):
    response = project_description_chain.invoke({"project_title": project_title,"degree_level":degree_level})
    return response

# Function to create a PDF from text using pdfkit
def create_pdf(details_str, file_name):
    pdfkit.from_string(details_str, file_name)


if "messages" not in st.session_state:
    st.session_state.messages = []


if uploaded_files and len(st.session_state.messages) == 0:
    doc_pages = []
    metadata = []
    summaries = []

    for file in uploaded_files:
        # st.markdown(f"preprocessing the file :{file}")
        st.success(f"Preprocessing file -> {file.name}")
        st.write(datetime.datetime.now())
        save_uploaded_file(file)
        file_path = os.path.join(os.getcwd(), "tempDir", file.name)
        loader = PyPDFLoader(file_path=str(file_path))
        pages = loader.load_and_split()
        doc_pages += pages
        summary = summarize_documents(pages)
        summaries.append(summary)
        # Extract and store metadata
        title, authors = extract_metadata(file_path)
        metadata.append({"title": title, "authors": authors, "pages": len(pages), "summary":summary['output_text']})
        st.success(f"Summary created for the file {file.name} successfully")
        st.write(datetime.datetime.now())
    delete_contents()

    llm, retriever = initialize_setup(doc_pages)    
    rag_chain = create_rag_chain(llm, retriever)
    rag_obj = create_rag_pipeline(rag_chain)

    st.session_state.rag_obj = rag_obj
    st.session_state.metadata = metadata
    st.session_state.messages.append({"role": "system", "context": "Initialization done!"})

# Display the metadata for each PDF
if "metadata" in st.session_state:
    for meta in st.session_state.metadata:
        st.write(f"**Title:** {meta['title']}")
        st.write(f"**Authors:** {meta['authors']}")
        st.write(f"**Number of Pages:** {meta['pages']}")
        st.write(f"**Summary:** {meta['summary']}")
        st.write("---")

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["context"])
    
    tab1, tab2, tab3 = st.tabs(["PDF Assistant", "ChatGPT","Project Idea generator"])

    with tab1:
        if uploaded_files:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["context"])

            if query := st.chat_input("Enter your query :-"):
                rag_obj = st.session_state.rag_obj
                with st.chat_message("user"):
                    st.markdown(query)
                    st.session_state.messages.append({"role": "user", "context": query})

                response = get_response(rag_obj, query)
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "context": response})
    
    with tab2:
    
        st.title("Chat with ChatGPT")
        
        # Initialize the ChatOpenAI object if not already done
        
        if "llm_chatgpt" not in st.session_state:
            st.session_state.llm_chatgpt = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv('OPENAI_API_KEY'))

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        
        
        # ChatGPT interaction
        # user_query = st.text_input("Ask me anything:", placeholder="Enter your question here")
        
        if chat_user_query := st.chat_input("Enter your query :-",key="tab2"):
            st.markdown("### Current Question:")
            with st.spinner('Thinking...'):
                llm = st.session_state.llm_chatgpt
                with st.chat_message("user"):
                        st.markdown(chat_user_query)
                st.session_state.chat_messages.append({"role": "user", "content": chat_user_query})
                
                # Ensure the input is passed as a string (not a dictionary)
                response = llm(st.session_state.chat_messages)  # This should pass a string to the model and return a valid response
                st.success("Response")
                with st.chat_message("assistant"):
                        st.markdown(response.content)
                st.session_state.chat_messages.append({"role": "assistant", "content": response.content})

    
    with tab3:
        st.title("Project Idea Generator")

        # Input fields
        degree_level = st.selectbox(
            "Select Degree Level:",
            ["Bachelor's", "Master's", "PhD", "Professional Certification"]
        )
        technology = st.text_input("Preferred Technology:")

        if st.button("Generate Project Ideas"):
            if degree_level and technology:
                with st.spinner("Generating project ideas..."):
                    st.markdown(degree_level)
                    ideas = generate_project_ideas(degree_level, technology)
                    st.markdown(ideas)
                    st.markdown(type(ideas))
                    ideas_list=ideas.split("\n")
                    st.markdown(ideas_list)
                    if ideas_list:
                        st.session_state.project_titles = ideas_list
                    else:
                        st.error("No project ideas generated.")

        #             if ideas:
        #                 project_titles = [idea['title'] for idea in ideas]
        #                 st.session_state.project_titles = project_titles
        #                 st.session_state.project_ideas = ideas
        #             else:
        #                 st.error("No project ideas generated.")
            else:
                st.error("Please provide both degree level and technology.")

        if 'project_titles' in st.session_state and st.session_state.project_titles:
            selected_title = st.selectbox("Select a Project Title:", st.session_state.project_titles)

            if st.button("Generate Project Details"):
                with st.spinner("Generating project details..."):
                    selected_idea = selected_title
                    if selected_idea:
                        details = generate_project_details(selected_title,degree_level)
                        st.markdown(details)

                        # output = format_chain.invoke({"details":details})
                        # st.write("The output format")
                        # st.markdown(output.content)
                        # st.write(type(output.content))
                        
                        # st.write(type(details))
                        details_str = f"**Project Title:** {selected_title}\n\n{details}"
                        file_name = f"{selected_title.replace(' ', '_')}_details.pdf"
                        create_pdf(details_str, file_name)

                        # Provide the PDF for download
                        with open(file_name, "rb") as pdf_file:
                            st.download_button(
                                label="Download Project Details as PDF",
                                data=pdf_file,
                                file_name=file_name,
                                mime="application/pdf"
                            )
                    else:
                        st.error("No details found for the selected project title.")
        else:
            st.info("Generate project ideas first to see project titles here.")




