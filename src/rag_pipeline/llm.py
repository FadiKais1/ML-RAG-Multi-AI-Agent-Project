from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

def get_llm():
    return ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

def generate_answer(question: str, context: str):
    llm = get_llm()

    prompt = f"""
    You are a helpful research assistant.
    Use the following context to answer the question.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    response = llm.invoke(prompt)
    return response.content
