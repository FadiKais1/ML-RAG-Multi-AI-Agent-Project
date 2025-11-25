from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str) -> List[str]:
    """
    Splits text into NVIDIA-safe chunks under 512 tokens.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    return splitter.split_text(text)
