from langchain_groq import ChatGroq
from .config import API_KEY

def get_llm(model_name:str, temperature:float, reasoning_effort:str,streaming:bool) -> ChatGroq:
    return ChatGroq(
        api_key=API_KEY,
        model=model_name,
        temperature=temperature,
        streaming=streaming,
        reasoning_effort=reasoning_effort
    )