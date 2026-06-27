# add_message
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field
from langchain_core.messages import BaseMessage



class ChatbotState(TypedDict):
    conversation: Annotated[list[BaseMessage],add_messages]