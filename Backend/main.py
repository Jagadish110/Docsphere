from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from bot import process_document, query_doc
import tempfile

app = FastAPI(title="RAG Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# store full text temporarily
DOCUMENT_TEXT = ""

@app.post("/upload")
async def upload_file(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    global DOCUMENT_TEXT
    try:
        if not file and not url:
            return {"error": "Either file or URL must be provided"}

        # process the file and extract text inside bot.py
        response, full_text = process_document(file=file, url=url, return_text=True)

        DOCUMENT_TEXT = full_text  # cache for later viewing
        return {"message": response, "preview": full_text[:] + "..."}  # send snippet to frontend

    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    question = request.question.strip()
    if not question:
        return {"error": "Question cannot be empty"}

    try:
        answer = query_doc(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/document")
async def view_document():
    """Return the entire loaded document text"""
    global DOCUMENT_TEXT
    if not DOCUMENT_TEXT:
        return {"error": "No document loaded yet"}
    return {"document": DOCUMENT_TEXT}

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running! Use /upload, /ask, and /document"}
