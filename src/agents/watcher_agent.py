import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.rag_pipeline.rag import index_pdf

DATA_DIR = "data"  # folder to watch for new PDFs


class PDFEventHandler(FileSystemEventHandler):
    """
    Watches the data directory for new or modified PDF files
    and indexes them into the RAG vector database.
    """

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            print(f"📄 Detected NEW PDF: {event.src_path}")
            index_pdf(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            print(f"✏️ Detected MODIFIED PDF: {event.src_path}")
            index_pdf(event.src_path)


def start_pdf_watcher():
    """
    Starts watching the DATA_DIR for PDF changes.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    event_handler = PDFEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path=DATA_DIR, recursive=False)

    print(f"👀 Watching folder for PDF changes: {os.path.abspath(DATA_DIR)}")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopping watcher...")
        observer.stop()

    observer.join()
