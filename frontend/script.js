let selectedDocumentId = null

async function askQuestion() {
    const question = document.getElementById("question").value.trim();
    const answerElement = document.getElementById("answer");
    const pdf_select = document.getElementById("pdfSelect")

    selectedDocumentId = pdf_select.value

    if (!question) {
        answerElement.innerText = "Please enter a question.";
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question, document_id : selectedDocumentId })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Request failed");
        }

        const sources = data.sources || []
        const sourceText = sources.length > 0 ? sources.map(source => 
            `${source.document.split(/[\\/]/).pop()} (Page ${source.page})`).join("\n") :
              "No sources available."

        answerElement.innerText = data.answer + "\n\nSources:\n" + sourceText;

    } catch (error) {
        answerElement.innerText = error.message;
    }
}

// upload pdf

async function uploadPDF(event) {
    if (event) event.preventDefault();

    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("uploadStatus");
    const pdf_select = document.getElementById("pdfSelect")
    const file = fileInput.files[0];

    if (!file) {
        status.textContent = "Please select a PDF first.";
        return;
    }

    status.style.display = "block";
    status.style.whiteSpace = "pre-line";
    status.style.color = "black";
    status.textContent = `Uploading ${file.name}...`;

    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("http://127.0.0.1:8000/upload-pdf", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        const option = document.createElement("option")
        option.value = data.document_id
        option.textContent = file.name
        pdf_select.appendChild(option)

        pdf_select.value = data.document_id
        selectedDocumentId = data.document_id

        status.style.color = "green";
        status.textContent =
            `File: ${data.filename} Chunks indexed: ${data["Total Chunks"]}`;

    } catch (error) {
        status.style.color = "red";
        status.textContent = `UPLOAD FAILED ${error.message}`;
    }
}

async function loadDocuments() {
    const pdfSelect = document.getElementById("pdfSelect");
    try{
        const response = await fetch("http://127.0.0.1:8000/documents")
        const documents = await response.json()
        console.log("Documents from backend:", documents);
        pdfSelect.innerHTML = '<option value="">Select a PDF</option>';
        documents.forEach(doc => {
            const option = document.createElement("option")
            option.value = doc.document_id
            option.textContent = doc.filename 
            pdfSelect.appendChild(option)
        });
    }catch (error) {
        console.error("Failed to load documents:", error);
    }
}
document.addEventListener("DOMContentLoaded", loadDocuments);
