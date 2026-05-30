import re
import math
from collections import Counter
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


# Lightweight semantic expansion for the POC.
# In production, this would be replaced by embeddings + pgvector.
SYNONYM_MAP = {
    "fraud": ["suspicious", "shell", "layering", "structuring", "aml", "sar"],
    "wire": ["transfer", "transfers", "payment", "payments", "outbound"],
    "company": ["entity", "entities", "corporation", "llc", "holdings"],
    "companies": ["entity", "entities", "corporation", "llc", "holdings"],
    "offshore": ["cayman", "bvi", "british", "virgin", "islands", "jurisdiction"],
    "owner": ["ownership", "beneficial", "ubo"],
    "owners": ["ownership", "beneficial", "ubo"],
    "cash": ["deposit", "deposits", "structured", "structuring", "ctr"],
    "transaction": ["transfer", "payment", "deposit", "activity"],
    "transactions": ["transfers", "payments", "deposits", "activity"],
}


def tokenize(text: str):
    """Convert text into searchable lowercase tokens."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def expand_query(query: str):
    """Expand the user query with simple domain-specific synonyms."""
    tokens = tokenize(query)
    expanded = set(tokens)

    for token in tokens:
        if token in SYNONYM_MAP:
            expanded.update(SYNONYM_MAP[token])

    return expanded


def create_chunks(text: str, chunk_size: int = 700, overlap: int = 120):
    """Split long evidence text into overlapping chunks for better retrieval."""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_embedding(text: str):
    """
    Create a lightweight vector embedding for the POC.

    This is not a production ML embedding model. It converts text into a
    weighted token vector so we can demonstrate vector-style retrieval without
    requiring external model downloads in the Citi sandbox.
    """
    tokens = tokenize(text)
    return Counter(tokens)


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two sparse token vectors."""
    if not vector_a or not vector_b:
        return 0.0

    shared_terms = set(vector_a.keys()).intersection(vector_b.keys())
    dot_product = sum(vector_a[term] * vector_b[term] for term in shared_terms)

    magnitude_a = math.sqrt(sum(value * value for value in vector_a.values()))
    magnitude_b = math.sqrt(sum(value * value for value in vector_b.values()))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def score_chunk(query: str, chunk: str, chunk_embedding=None):
    """Score a chunk using vector similarity + exact phrase bonus."""
    query_lower = query.lower().strip()
    chunk_lower = chunk.lower()

    expanded_query_tokens = expand_query(query)
    query_embedding = Counter(expanded_query_tokens)

    if chunk_embedding is None:
        chunk_embedding = create_embedding(chunk)

    matched_terms = sorted(set(query_embedding.keys()).intersection(set(chunk_embedding.keys())))
    vector_score = cosine_similarity(query_embedding, chunk_embedding)

    # Small bonus when the exact phrase appears in the chunk.
    exact_match_bonus = 0.25 if query_lower and query_lower in chunk_lower else 0.0

    final_score = min(vector_score + exact_match_bonus, 1.0)

    return round(final_score, 3), matched_terms


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


# New endpoint to get a document by filename
@app.get("/api/v1/document/{filename}")
def get_document(filename: str):
    for document in DOCUMENTS:
        if document["filename"] == filename:
            return {
                "file_name": document["filename"],
                "content_type": document["content_type"],
                "size_bytes": document["size_bytes"],
                "text": document["text"]
            }

    return {
        "error": "Document not found",
        "file_name": filename
    }


@app.post("/api/v1/ingest")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()

    # For this first version, we support text-based files.
    # Later, we can add PDF, DOCX, and CSV-specific extraction.
    text = content.decode("utf-8", errors="ignore")

    chunks = create_chunks(text)
    chunk_embeddings = [create_embedding(chunk) for chunk in chunks]

    document = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "text": text,
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings
    }

    DOCUMENTS.append(document)

    return {
        "message": "File ingested successfully",
        "file_name": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "stored_documents": len(DOCUMENTS),
        "chunks_created": len(chunks),
        "next_step": "Lightweight vector search is now available. Production semantic search would use model embeddings with pgvector."
    }


@app.post("/api/v1/search")
def search_evidence(request: SearchRequest):
    query = request.query.strip()

    if not query:
        return {
            "query": request.query,
            "top_k": request.top_k,
            "documents_indexed": len(DOCUMENTS),
            "results_found": 0,
            "search_type": "lightweight_vector_similarity_with_query_expansion",
            "results": []
        }

    scored_results = []

    for document in DOCUMENTS:
        chunks = document.get("chunks") or create_chunks(document["text"])
        chunk_embeddings = document.get("chunk_embeddings") or [create_embedding(chunk) for chunk in chunks]

        for chunk_index, chunk in enumerate(chunks):
            chunk_embedding = chunk_embeddings[chunk_index]
            score, matched_terms = score_chunk(query, chunk, chunk_embedding)

            if score > 0:
                scored_results.append({
                    "document_name": document["filename"],
                    "case_id": "CASE-2024-001",
                    "score": score,
                    "text": chunk,
                    "source": "uploaded document",
                    "chunk_index": chunk_index,
                    "matched_terms": matched_terms,
                    "retrieval_method": "cosine_similarity_over_lightweight_vectors"
                })

    scored_results.sort(key=lambda result: result["score"], reverse=True)
    scored_results = scored_results[:max(request.top_k, 1)]

    results = []
    for index, result in enumerate(scored_results, start=1):
        results.append({
            "rank": index,
            **result
        })

    return {
        "query": request.query,
        "top_k": request.top_k,
        "documents_indexed": len(DOCUMENTS),
        "results_found": len(results),
        "search_type": "lightweight_vector_similarity_with_query_expansion",
        "results": results
    }