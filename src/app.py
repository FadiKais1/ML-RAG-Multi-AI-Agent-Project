from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline.rag import answer_question
from src.agents.watcher_agent import start_pdf_watcher
import threading

app = FastAPI(title="Dynamic RAG Agent API")

# -----------------------------
# Models
# -----------------------------
class Question(BaseModel):
    question: str


# -----------------------------
# Routes
# -----------------------------
@app.post("/ask")
def ask_question(item: Question):
    """
    Main chatbot endpoint.
    """
    question = item.question
    answer = answer_question(question)
    return {"question": question, "answer": answer}


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


# -----------------------------
# Background: Start Updating Agent
# -----------------------------
def start_agent_background():
    watcher_thread = threading.Thread(target=start_pdf_watcher, daemon=True)
    watcher_thread.start()

start_agent_background()
