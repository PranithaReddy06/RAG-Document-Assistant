from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.vectorstore import vectorstore
import hashlib
from fastapi import HTTPException
import os

def load_and_split_pdf(filepath:str):
    try:
        file = PyPDFLoader(filepath)
        document = file.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read PDF. The file may be corrupted or invalid.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks =  text_splitter.split_documents(document)
    
    with open(filepath, "rb") as PDF_file:
        document_id = hashlib.sha256(PDF_file.read()).hexdigest()
    
    for chunk in chunks:
        chunk.metadata["source"] = filepath
        chunk.metadata["document_id"] = document_id
    
    return chunks

def index_pdf(filepath:str):
    try:
        with open(filepath, "rb") as PDF_file:
            document_id = hashlib.sha256(PDF_file.read()).hexdigest()
    
        existing_pdf = vectorstore.get(where={"document_id": document_id})
        
        if existing_pdf["ids"]:
            print("This pdf is already indexed")
            return 0
    
        chunks = load_and_split_pdf(filepath)
        chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text found in pdf")

        try:
            vectorstore.add_documents(chunks)
        except Exception:
            raise HTTPException(status_code=500,detail="Failed to generate embeddings or store document")

        return(len(chunks))
    
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise