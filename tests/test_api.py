from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import Mock
from unittest.mock import patch
from app.rag.service import RAGService

client = TestClient(app)
client = TestClient(app, raise_server_exceptions=False)

def test_get_documents():
    response = client.get("/documents")
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert "application/json" in response.headers["content-type"]

def test_ask_question_validation():
    response = client.post("/ask", json={"question":"Hi"})
    assert response.status_code == 422

def test_ask_question_missing_question():
    response = client.post("/ask", json={"document_id":"test-doc-123"})
    assert response.status_code == 422

def test_ask_question_too_short():
    response = client.post("/ask", json={"question":"Hi", "document_id":"test-doc-123"})
    assert response.status_code == 422

def test_ask_question_missing_document_id():
    response = client.post("/ask", json={"question":"What skills are listed?"})
    assert response.status_code == 422

def test_ask_response_isjson():
    response = client.post("/ask", json={"question":"What skills are listed?", "document_id":"test-doc-123"})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

def test_ask_response_has_answers_and_sources():
    response = client.post("/ask", json={"question":"What skills are listed?", "document_id":"test-doc-123"})
    
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    
def test_ask_question_empty_document_id():
    response = client.post("/ask", json={"question":"What skills are listed?", "document_id":""})
    assert response.status_code == 422

def test_get_documents_have_required_fields():
    response = client.get("/documents")
    assert response.status_code == 200
    
    documents = response.json()
    for doc in documents:
        assert "document_id" in doc
        assert "filename" in doc

def test_uploadPDF_rejects_nopdf():
    response = client.post("/upload-pdf", files={"file": ("notes.txt", b"not a PDF", "text/plain")})
    assert response.status_code != 200
    
def test_uploadPDF_rejects_emptyfile():
    response = client.post("/upload-pdf", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code != 200

def test_uploadPDF_rejects_invalidPDF():
    response = client.post("/upload-pdf", files={"file": ("fake.pdf", b"This is not real PDF", "application/pdf")})
    assert response.status_code != 200

def test_uploadPDF_rejects_wrong_extension():
    response = client.post("/upload-pdf", files={"file": ("notes.txt", b"%PDF-1.4 fake content", "application/pdf")})
    assert response.status_code != 200
    
def test_uploadPDF_rejects_oversized_file():
    large_content = b"%PDF-" + b"A" * (10 * 1024 * 1024)

    response = client.post("/upload-pdf", files={"file":("large.pdf", large_content, "application/pdf")})
    assert response.status_code != 200

def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question":"", "document_id":"test-doc-123"})
    assert response.status_code == 422

def test_ask_question_too_long():
    response = client.post("/ask", json={"question":"A"*1001, "document_id":"test-doc-123"})
    assert response.status_code == 422

def test_ask_question_missing_body():
    response = client.post("/ask")
    assert response.status_code == 422

def test_uploadPDF_missing_fileF():
    response = client.post("/upload-pdf")
    assert response.status_code == 422

def test_get_document_items_are_dictonaries():
    response = client.get("/documents")
    assert response.status_code == 200
    
    documents = response.json()
    for doc in documents:
        assert isinstance(doc, dict)

def test_document_rejects_post():
    response = client.post("/documents")
    assert response.status_code == 405

def test_ask_rejects_get():
    response = client.get("/ask")
    assert response.status_code == 405

def test_upload_pdf_rejects_get():
    response = client.get("/upload-pdf")
    assert response.status_code == 405

def test_ask_when_no_chunks_found():
    from unittest.mock import patch
    
    with patch("app.main.rag_service.askQuestion") as mock_ask:
        mock_ask.return_value = {
            "answer": "I could not find that information in the document.",
            "sources": []
        }
        response = client.post("/ask", json={"question":"What skills are listed?", "document_id":"test-doc-123"})
    assert response.status_code == 200
    assert response.json()["sources"] == []

def test_rag_service_returns_fallback_when_no_chunks():
    service = object.__new__(RAGService)
    service.vectorstore = Mock()
    service.vectorstore.similarity_search_with_score.return_value = []
    
    result = service.askQuestion("what are my skills?", document_id="test-doc-123")
    assert result == {
        "answer": "I could not find that information in the document.",
        "sources": []
    }

def test_rag_service_filters_by_documentid():
    from unittest.mock import Mock
    from app.rag.service import RAGService
    
    service = object.__new__(RAGService)
    service.vectorstore = Mock()
    service.vectorstore.similarity_search_with_score.return_value = []
    
    service.askQuestion("what are my skills?", document_id="test-doc-123")
    
    service.vectorstore.similarity_search_with_score.assert_called_once_with(
        "what are my skills?", k=8, filter = {"document_id":"test-doc-123"}        
    )

def test_upload_pdf_success():
    with patch("app.main.index_pdf", return_value = 3),\
        patch("app.main.vectorstore.get", return_value = {"ids":[]}):
        response = client.post("upload-pdf", files={"file":("test.pdf", b"%PDF-1.4 test content", "application/pdf")})
    
    assert response.status_code == 200
    assert response.json()["Total Chunks"] == 3
    assert response.json()["message"] == "PDF uploaded and indexed successfully."
    
def test_upload_pdf_rejects_duplicate_pdf():
    with patch("app.main.vectorstore.get", return_value = {"ids":["existing-chunk-1"]}):
        response = client.post("upload-pdf", files={"file":("test.pdf", b"%PDF-1.4 test content", "application/pdf")})
        
    assert response.status_code == 409
    assert response.json()["detail"] == "This PDF has already been uploaded and indexed."

def test_get_documents_returns_document_details():
    with patch("app.main.vectorstore.get",
                return_value = {"metadatas":[{"document_id":"pdf-123",
                                              "source":"documents/uploads/test.pdf"}]}):
        response = client.get("/documents")
        
    assert response.status_code == 200
    assert response.json() == [{"document_id": "pdf-123","filename": "test.pdf"}]

def test_upload_pdf_indexing_failure():
    with patch("app.main.vectorstore.get", return_value={"ids": []}), \
         patch("app.main.index_pdf", side_effect=Exception("Indexing failed")):

        response = client.post("/upload-pdf",files={"file": ("test.pdf",b"%PDF-1.4 test content","application/pdf")})

    assert response.status_code == 500

def test_ask_resonse_isjson():
    with patch("app.main.rag_service.askQuestion") as mock_ask:
        mock_ask.return_value = {"answer":"python is listed", "sources":[]}
        
        response = client.post("/ask", json={"question": "What skills are listed?", "document_id": "test-doc-123"})
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"] 

def test_ask_resonse_has_question_and_answers():
    with patch("app.main.rag_service.askQuestion") as mock_ask:
        mock_ask.return_value = {"answer":"python is listed", "sources":[]}
        
        response = client.post("/ask", json={"question": "What skills are listed?", "document_id": "test-doc-123"})
        
        data = response.json()
        assert response.status_code == 200
        assert "answer" in data
        assert "sources" in data

def test_upload_pdf_indexing_failure():
    with patch("app.main.vectorstore.get", return_value={"ids": []}), \
         patch("app.main.index_pdf", side_effect=Exception("Indexing failed")):

        response = client.post(
            "/upload-pdf",
            files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
        )

    assert response.status_code == 500
    assert "Failed to process and index PDF" in response.json()["detail"]

def test_rag_service_handles_ollama_failure():
    from langchain_core.documents import Document

    service = object.__new__(RAGService)

    service.vectorstore = Mock()
    service.vectorstore.similarity_search_with_score.return_value = [
        (Document(
            page_content="Python is a programming language.",
            metadata={"source": "test.pdf", "page": 0}
        ), 0.5)
    ]

    service.ranker = Mock()
    service.ranker.rerank.return_value = [
        {
            "text": "Python is a programming language.",
            "meta": {"source": "test.pdf", "page": 0},
            "score": 0.9
        }
    ]

    service.llm = Mock()
    service.llm.invoke.side_effect = Exception("Ollama unavailable")

    result = service.askQuestion("What is Python?")

    assert result["answer"] == "The AI model is currently unavailable. Please try again later."
    assert result["sources"] == []