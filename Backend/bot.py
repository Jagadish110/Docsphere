import os
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing_extensions import TypedDict
import tempfile


class mystate(TypedDict):
    question:str

load_dotenv()
Groq_api_key = os.getenv("groq_api_key")

retriever_chain = None
def process_document(file=None, url=None,return_text=False):
    global retriever_chain
    if not file and not url:
        raise ValueError("Please upload a file or provide a URL")

    # Load document
    if file:
        suffix = file.filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            documents = PyPDFLoader(tmp_path).load()
        elif suffix == "txt":
            documents = TextLoader(tmp_path).load()
        elif suffix == "pptx":
            documents = UnstructuredPowerPointLoader(tmp_path).load()
        elif suffix == "xlsx":
            documents = UnstructuredExcelLoader(tmp_path).load()
        elif suffix in ["docx", "doc"]:
            documents = UnstructuredWordDocumentLoader(tmp_path).load()
        else:
            raise ValueError("Unsupported file type")

    elif url:
        documents = WebBaseLoader(url).load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    full_text = "\n\n".join([chunk.page_content for chunk in chunks])

    # Create embeddings & retriever
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vector_db = Chroma.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Define prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful professor. Use the given context to answer questions precisely."),
        ("human", """Answer strictly based on the context:
<context>
{context}
</context>
Question: {input}""")
    ])

    llm = ChatGroq(api_key=Groq_api_key, model="llama-3.1-8b-instant")
   # === CORRECT RAG CHAIN FOR LANGCHAIN 1.0.3 ===
    rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
    retriever_chain = rag_chain
    if return_text:
        return "Document processed successfully", full_text
    return "Document processed successfully"


def query_doc(question:str):
    global retriever_chain
    if retriever_chain is None:
        raise ValueError("No document has been processed yet. Please upload a document first.")
    result = retriever_chain.invoke(question)
    return result
 
