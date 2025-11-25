import uuid
from typing import List, Dict
import chromadb
from chromadb import PersistentClient

DB_DIR = "models/chroma_db"
COLLECTION_NAME = "documents"


def get_chroma_client():
    """
    Uses the NEW Chroma architecture (PersistentClient).
    No deprecated Settings() usage.
    """
    client = PersistentClient(path=DB_DIR)
    return client


def get_collection():
    """
    Gets or creates the Chroma collection for embeddings.
    """
    client = get_chroma_client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # vector similarity metric
        )
    return collection


def add_documents(chunks: List[str], embeddings: List[List[float]], metadata: Dict = None):
    """
    Adds embedded chunks to Chroma with unique IDs.
    """
    collection = get_collection()

    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[metadata or {} for _ in chunks]
    )

    print(f"✅ Added {len(chunks)} chunks to Chroma collection '{COLLECTION_NAME}'")


def query_similar(query_embedding: List[float], k: int = 3):
    """
    Performs vector similarity search using the new API.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results
