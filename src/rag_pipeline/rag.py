from .loader import load_pdf
from .chunker import chunk_text
from .embeddings import get_embeddings_model, embed_chunks
from .vector_db import add_documents, query_similar
from .llm import generate_answer


def index_pdf(file_path: str):
    """
    Loads a PDF, chunks it, embeds it using NVIDIA,
    and stores it in the vector database.
    """
    print(f"📄 Loading PDF: {file_path}")
    text = load_pdf(file_path)

    if not text.strip():
        print("❌ PDF contains no extractable text.")
        return

    print("🧩 Chunking text...")
    chunks = chunk_text(text)

    if len(chunks) == 0:
        print("❌ No chunks generated.")
        return

    print(f"📦 Total chunks: {len(chunks)}")

    print("🔢 Generating embeddings (NVIDIA)...")
    embeddings = embed_chunks(chunks)

    if len(embeddings) != len(chunks):
        print("❌ Mismatch: embeddings != chunks!")
        return

    print("🗃 Saving to vector DB...")
    add_documents(chunks, embeddings, metadata={"source": file_path})

    print("🎉 PDF indexed successfully!")


def answer_question(question: str, k: int = 3) -> str:
    """
    Full RAG pipeline:
    1. Embed question
    2. Retrieve top chunks
    3. Build context
    4. Ask LLM for answer
    """

    print("❓ Embedding user question (NVIDIA)...")
    embed_model = get_embeddings_model()
    query_embedding = embed_model.embed_query(question)

    print("🔍 Searching similar chunks in vector database...")
    results = query_similar(query_embedding, k=k)

    if "documents" not in results or len(results["documents"]) == 0:
        return "⚠️ No relevant information found in the database."

    docs = results["documents"][0]
    context = "\n\n".join(docs)

    print("💬 Generating answer using LLM...")
    answer = generate_answer(question, context)

    return answer
