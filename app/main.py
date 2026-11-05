from contextlib import asynccontextmanager

from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.rag.service import RAGService
from app.schemas.question import QuestionSchema
from app.rag.vectorstore import vectorstore
from app.rag.ingestion import index_pdf
import os
import uuid
import hashlib
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@asynccontextmanager
async def lifespan(app:FastAPI):
    yield
    logger.info("Closing ChromaDB client...")
    vectorstore._client.close()
    logger.info("ChromaDB client closed successfully.")

app = FastAPI(lifespan=lifespan)

# Centralized exception handling
@app.exception_handler(Exception)
def global_exceptio_handler(request,exc):
    logger.exception("An unexcepted error occured")
    return JSONResponse(status_code=500, content={"detail":"An unexpected server error occurred."})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService(vectorstore)

UPLOAD_DIR = "documents/uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024 
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ask")
def ask_question(request : QuestionSchema):
    logger.info("Received question from user")
    response = rag_service.askQuestion(request.question, document_id=request.document_id)
    return response
    
@app.post("/upload-pdf")
def uploadPDF(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_content = file.file.read()
    
    if not file_content.startswith(b"%PDF-"):
        raise HTTPException(status_code=404, detail="Invalid PDF file")
    
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="PDF file size must not exceed 10 MB.")
    
    document_id = hashlib.sha256(file_content).hexdigest()
    existing_pdf = vectorstore.get(where={"document_id":document_id})
    if existing_pdf["ids"]:
        raise HTTPException(status_code=409, detail="This PDF has already been uploaded and indexed.")
    
    safe_filename = os.path.basename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{safe_filename}"
    filepath = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open (filepath, "wb") as pdf_file:
        pdf_file.write(file_content)
        
    try:
        logger.info("PDF indexing started: %s", safe_filename)
        total_chunks = index_pdf(filepath)
        logger.info("PDF indexing completed: %s | Chunks: %s", safe_filename, total_chunks)
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail="Failed to process and index PDF")
    
    return {"message": "PDF uploaded and indexed successfully.",
            "filename":unique_filename,
            "document_id":document_id,
            "Total Chunks":total_chunks}
    
@app.get("/documents")
def get_documents():
    data = vectorstore.get(include=["metadatas"])
    documents = {}
    
    for metadata in data["metadatas"]:
        doc_id = metadata.get("document_id")
        filename = os.path.basename(metadata.get("source",""))
        
        if doc_id:
            documents[doc_id] = {"document_id":doc_id, "filename":filename}
        
    return list(documents.values())
    