from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from storage import DOCUMENTS

app = FastAPI(
    title="IMS AI Search API",
    description="Python backend for the IMS AI Search prototype",
    version="0.1.0"
)

# This allows the HTML frontend to call the backend while developing locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "ok",
        "service": "IMS AI Search API",
        "version": "0.1.0"
    }


# New endpoint to list documents
@app.get("/api/v1/documents")
def get_documents():
    return {
        "count": len(DOCUMENTS),
        "documents": [
            {
                "file_name": document["filename"],
                "content_type": document["content_type"],
                "size_bytes": document["size_bytes"]
            }
            for document in DOCUMENTS
        ]
    }


@app.post("/api/v1/ingest")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()

    # For this first version, we support text-based files.
    # Later, we can add PDF, DOCX, and CSV-specific extraction.
    text = content.decode("utf-8", errors="ignore")

    document = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "text": text
    }

    DOCUMENTS.append(document)

    return {
        "message": "File ingested successfully",
        "file_name": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "stored_documents": len(DOCUMENTS),
        "next_step": "Keyword search is now available. Semantic embeddings will be added next."
    }


@app.post("/api/v1/search")
def search_evidence(request: SearchRequest):
    query = request.query.lower().strip()

    if not query:
        return {
            "query": request.query,
            "top_k": request.top_k,
            "documents_indexed": len(DOCUMENTS),
            "results_found": 0,
            "results": []
        }

    results = []

    for document in DOCUMENTS:
        text = document["text"]
        text_lower = text.lower()

        if query in text_lower:
            start_index = text_lower.find(query)
            snippet_start = max(start_index - 120, 0)
            snippet_end = min(start_index + len(query) + 250, len(text))
            snippet = text[snippet_start:snippet_end]

            results.append({
                "rank": len(results) + 1,
                "document_name": document["filename"],
                "case_id": "CASE-POC",
                "score": 1.0,
                "text": snippet,
                "source": "uploaded document"
            })

    # Limit the number of returned results based on top_k.
    results = results[:max(request.top_k, 1)]

    return {
        "query": request.query,
        "top_k": request.top_k,
        "documents_indexed": len(DOCUMENTS),
        "results_found": len(results),
        "results": results
    }