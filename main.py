import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO
import pdfplumber  # For PDF processing
from docx import Document  # For DOCX processing

# LLM and key loading function
def load_LLM(openai_api_key):
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0.2, openai_api_key=openai_api_key)
    return llm

# Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Intro: instructions
col1, col2 = st.columns(2)
with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")
with col2:
    st.write("First project by gentlebear courtesy AI Accelera")

# Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key ", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()

# Input
st.markdown("## Upload the file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

# Output
st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    file_input = ""
    file_type = uploaded_file.name.split('.')[-1].lower()

    # Process TXT file
    if file_type == "txt":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        file_input = stringio.read()

    # Process PDF file
    elif file_type == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            file_input = "\n".join(page.extract_text() for page in pdf.pages)

    # Process DOCX file
    elif file_type == "docx":
        doc = Document(uploaded_file)
        file_input = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20000 words.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning('Please insert OpenAI API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=5000, 
            chunk_overlap=350
        )

        splitted_documents = text_splitter.create_documents([file_input])

        llm = load_LLM(openai_api_key=openai_api_key)

        summarize_chain = load_summarize_chain(
            llm=llm, 
            chain_type="map_reduce"
        )

        summary_output = summarize_chain.run(splitted_documents)

        st.write(summary_output)
