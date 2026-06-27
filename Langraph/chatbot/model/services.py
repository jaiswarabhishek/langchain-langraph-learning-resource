from llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from system_prompt import SYSTEM_PROMPT
from pydantic import BaseModel,Field
from langchain_core.messages import AIMessage
from states.chatbot_state import ChatbotState

def ask_llm(state:ChatbotState, model_name:str, temperature:float, reasoning_effort:str,streaming:bool) -> AIMessage:
    llm = get_llm(model_name, temperature, reasoning_effort,streaming)

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", SYSTEM_PROMPT),
    #     ("human", "{query}")
    # ])

    response = llm.invoke(state["conversation"])

    return {"conversation": [AIMessage(content=response)]}