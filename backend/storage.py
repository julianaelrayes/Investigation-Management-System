from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary in-memory document storage for the prototype.
# This will reset every time the server restarts.
# In production, this would be replaced by PostgreSQL + pgvector.
DOCUMENTS = []