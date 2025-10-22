import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
Groq_api_key = os.environ['groq_api_key']

# UI Header
st.markdown("<h1 style='text-align: center;'>DocSphere</h1>", unsafe_allow_html=True)
st.markdown("<div class='title'>Upload your documents or URLs and ask questions with ease!</div>", unsafe_allow_html=True)

# Session State Setup
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'source' not in st.session_state:
    st.session_state.source = None

# --- File Upload Options ---
source_option = st.selectbox(
    'Add Files',
    ('PDF', 'Website Link', 'TextFile', 'PowerPoint', 'ExcelFile'),
    help="Choose the type of source you want to upload or input."
)

documents = []
current_source = None

if source_option == 'PDF':
    uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf'])
    if uploaded_file:
        with open('temp.pdf', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader('temp.pdf')
        documents = loader.load()
        current_source = uploaded_file.name

elif source_option == 'Website Link':
    url = st.text_input('Paste the URL..', placeholder="https://example.com")
    if url:
        loader = WebBaseLoader(url)
        documents = loader.load()
        current_source = url

elif source_option == 'TextFile':
    uploaded_file = st.file_uploader('Upload a Text file', type=['txt'])
    if uploaded_file:
        with open('temp.txt', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        loader = TextLoader('temp.txt')
        documents = loader.load()
        current_source = uploaded_file.name

elif source_option == 'PowerPoint':
    uploaded_file = st.file_uploader('Upload a PowerPoint file', type=['pptx'])
    if uploaded_file:
        with open('temp.pptx', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        loader = UnstructuredPowerPointLoader('temp.pptx')
        documents = loader.load()
        current_source = uploaded_file.name

elif source_option == 'ExcelFile':
    uploaded_file = st.file_uploader('Upload an Excel file', type=['xlsx'])
    if uploaded_file:
        with open('temp.xlsx', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        loader = UnstructuredExcelLoader('temp.xlsx')
        documents = loader.load()
        current_source = uploaded_file.name


# --- Process the Document ---
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if st.session_state.source != current_source:
        document_chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        st.session_state.vector_db = FAISS.from_documents(document_chunks, embeddings)
        st.session_state.source = current_source

    # Combine all extracted text
    full_text = "\n\n".join([doc.page_content for doc in documents])

    # --- ✅ SIDEBAR CONTENT ---
    with st.sidebar:
        st.markdown("### 📁 Uploaded Document Info")
        st.success(f"**File:** {current_source}")
        st.write(f"**Document Type:** {source_option}")
        st.markdown("---")

        # Limit to first 3000 characters for performance
        preview_text = full_text[:]
        st.markdown("### 📄 Extracted Content Preview")
        st.text_area("Document Text", preview_text, height=400)
        if len(full_text) > 3000:
            st.info("Showing first 3000 characters only (for preview).")

        st.markdown("---")
        st.write("You can now ask questions based on this document below 👇")

# --- Setup RAG Components ---
vector_db = st.session_state.vector_db

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professor and helpful assistant. Use the context to answer questions."),
    ("human", """Answer the question based on the context only.
<context>
{context}
</context>
Question: {input}""")
])

llm = ChatGroq(api_key=Groq_api_key, model='llama-3.1-8b-instant')
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

if vector_db:
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    retriever_chain = create_retrieval_chain(retriever, document_chain)
else:
    st.warning("Please upload a file or provide a URL first.")
# Custom CSS for Aurora Lights theme
st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:wght@300;400;600;700&display=swap">
    <style>
        .stApp {
            font-family: 'DM Serif Display', serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            color: #ffffff;
        }
        .stChatMessage {
            padding: 1.25rem;
            border-radius: 1rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(25px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        }
        .stChatMessage.user {
            background: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #a8e6cf;
        }
        .stChatMessage.assistant {
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid #56ab2f;
        }
        .stChatInput input {
            border-radius: 2rem !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            padding: 1rem 1.5rem !important;
            font-family: 'DM Serif Display', serif !important;
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            backdrop-filter: blur(25px);
            transition: all 0.3s ease;
        }
        .stChatInput input:focus {
            border-color: #a8e6cf !important;
            box-shadow: 0 0 0 3px rgba(168, 230, 207, 0.3) !important;
            background: rgba(255, 255, 255, 0.15);
        }
        h1 {
            font-weight: 700;
            font-size: 3rem;
            background: linear-gradient(135deg, #ffffff 0%, #ffd93d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .stMarkdown {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #ffffff;
        }
        .stSelectbox > div > div > div {
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(25px);
        }
        .stFileUploader > div > div > div {
            border-radius: 1rem;
            border: 2px dashed rgba(168, 230, 207, 0.4);
            background: rgba(168, 230, 207, 0.1);
            backdrop-filter: blur(25px);
        }
        .stTextInput > div > div > input {
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1rem 1.25rem;
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            backdrop-filter: blur(25px);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
            backdrop-filter: blur(25px);
            border-right: 1px solid rgba(255, 255, 255, 0.15);
        }
        .stSuccess > div {
            background: rgba(86, 171, 47, 0.2);
            border: 1px solid rgba(86, 171, 47, 0.5);
            color: #ffffff;
            border-radius: 1rem;
        }
        .stTextArea > div > div > textarea {
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            backdrop-filter: blur(25px);
        }
    </style>
""", unsafe_allow_html=True)

# --- User Query Section ---
if prompt:=st.chat_input("Ask Anything .."):

    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
              
        try:

            with st.spinner('Generating the output...'):
             response = retriever_chain.invoke({'input': prompt})
             st.write(response['answer'])
        except Exception as e:
    
         st.error("You should only ask question related to your documents.")
