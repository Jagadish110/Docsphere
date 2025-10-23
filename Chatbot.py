import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
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
## Custom CSS for a unique "Nebula" theme: Cosmic gradients, glassmorphism effects, and a futuristic vibe
st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&display=swap">
    <style>
        .stApp {
            font-family: 'Manrope', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e6ed;
        }
        .stChatMessage {
            padding: 1.25rem;
            border-radius: 1rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .stChatMessage.user {
            background: rgba(99, 102, 241, 0.15);
            border-left: 4px solid #6366f1;
        }
        .stChatMessage.assistant {
            background: rgba(236, 72, 153, 0.15);
            border-left: 4px solid #ec4899;
        }
        .stChatInput input {
            border-radius: 1.5rem !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            padding: 1rem 1.25rem !important;
            font-family: 'Manrope', sans-serif !important;
            background: rgba(255, 255, 255, 0.05);
            color: #e0e6ed;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .stChatInput input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
            background: rgba(255, 255, 255, 0.1);
        }
        h1 {
            font-weight: 700;
            font-size: 3rem;
            background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .stMarkdown {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #e0e6ed;
        }
        .stSelectbox > div > div > div {
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        }
        .stFileUploader > div > div > div {
            border-radius: 1rem;
            border: 2px dashed rgba(99, 102, 241, 0.3);
            background: rgba(99, 102, 241, 0.05);
            backdrop-filter: blur(10px);
        }
        .stTextInput > div > div > input {
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1rem 1.25rem;
            background: rgba(255, 255, 255, 0.05);
            color: #e0e6ed;
            backdrop-filter: blur(10px);
        }
        .stTextInput > div > div > input:focus {
            border-color: #ec4899;
            box-shadow: 0 0 0 3px rgba(236, 72, 153, 0.2);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stSuccess > div {
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid rgba(34, 197, 94, 0.5);
            color: #e0e6ed;
            border-radius: 1rem;
        }
        .stWarning > div {
            background: rgba(245, 158, 11, 0.2);
            border: 1px solid rgba(245, 158, 11, 0.5);
            color: #e0e6ed;
            border-radius: 1rem;
        }
        .stTextArea > div > div > textarea {
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.05);
            color: #e0e6ed;
            backdrop-filter: blur(10px);
        }
        .stSpinner > div {
            color: #6366f1;
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
