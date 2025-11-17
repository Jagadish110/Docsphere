from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Groq_api_key = os.getenv("groq_api_key")

# Global memory (safe for Render)
VECTOR_STORE = None
RETRIEVER_CHAIN = None
DOCUMENT_TEXT = ""

class QueryRequest(BaseModel):
    question: str


# -------------------------------------------------------
# Document Processing
# -------------------------------------------------------
def load_document(file=None, url=None):
    if file:
        suffix = file.filename.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            docs = PyPDFLoader(tmp_path).load()
        elif suffix == "txt":
            docs = TextLoader(tmp_path).load()
        elif suffix == "pptx":
            docs = UnstructuredPowerPointLoader(tmp_path).load()
        elif suffix == "xlsx":
            docs = UnstructuredExcelLoader(tmp_path).load()
        elif suffix in ["docx", "doc"]:
            docs = UnstructuredWordDocumentLoader(tmp_path).load()
        else:
            raise ValueError("Unsupported file type")

        return docs

    elif url:
        return WebBaseLoader(url).load()

    else:
        raise ValueError("No file or URL provided")


def build_rag_chain(chunks):
    global VECTOR_STORE

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Use FAISS instead of Chroma → safe for Render
    VECTOR_STORE = FAISS.from_documents(chunks, embeddings)
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful professor. Answer ONLY using the provided context."),
        ("human",
         """Use ONLY this context to answer:
<context>
{context}
</context>

Question: {input}
"""),
    ])

    llm = ChatGroq(api_key=Groq_api_key, model="llama-3.1-8b-instant")

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.post("/upload")
async def upload_file(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    global RETRIEVER_CHAIN, DOCUMENT_TEXT

    try:
        docs = load_document(file=file, url=url)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        DOCUMENT_TEXT = "\n\n".join([c.page_content for c in chunks])
        RETRIEVER_CHAIN = build_rag_chain(chunks)

        return {
            "message": "Document processed successfully",
            "preview": DOCUMENT_TEXT[:500] + "..."
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/ask")
async def ask_question(request: QueryRequest):
    global RETRIEVER_CHAIN

    if RETRIEVER_CHAIN is None:
        return {"error": "No document uploaded"}

    try:
        answer = RETRIEVER_CHAIN.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/document")
async def view_document():
    if not DOCUMENT_TEXT:
        return {"error": "No document loaded yet"}
    return {"document": DOCUMENT_TEXT}


@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API is running",
        "routes": ["/upload", "/ask", "/document"]
    }

@app.head("/")
async def root_head():
    return
