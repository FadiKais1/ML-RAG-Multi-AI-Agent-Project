from dotenv import load_dotenv
load_dotenv()

import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

def get_embeddings_model():
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

def embed_chunks(chunks):
    model = get_embeddings_model()
    embeddings = model.embed_documents(chunks)
    return embeddings
