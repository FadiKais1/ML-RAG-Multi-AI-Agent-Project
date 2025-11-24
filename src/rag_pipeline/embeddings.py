from langchain_openai import OpenAIEmbeddings
from typing import List

def get_embeddings_model():
    """
    Returns the embeddings model used for generating vector representations.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")

def embed_chunks(chunks: List[str]):
    """
    Takes a list of text chunks and returns a list of embeddings.
    """
    model = get_embeddings_model()
    embeddings = model.embed_documents(chunks)
    return embeddings
