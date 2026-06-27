from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from pydantic import BaseModel,Field
import operator
from langchain_core.messages import (HumanMessage,SystemMessage,AIMessage,BaseMessage)
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sqlite3
# add_message
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun,GoogleSerperRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper,GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

#RAG Tool
global retriever 

def process_pdf(file_path):
    global retriever
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory="chroma_db")
    retriever = vectorstore.as_retriever()

# tools
api_wrapper_wikki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper_wikki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# search = DuckDuckGoSearchRun(name="DuckDuckGoSearch")
google_serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

@tool
def search_tool(query: str):
    """Use this tool to search for real-time information on the web."""
    return google_serper.run(query)

@tool
def pdf_rag_tool(query: str):
    """Use this tool to answer questions from the uploaded PDF document."""
    global retriever
    if 'retriever' not in globals():
        return "No PDF has been uploaded yet."
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

# make tool list

tool_list = [wiki_tool,arxiv_tool,search_tool,pdf_rag_tool]

#llama-3.3-70b-versatile
model = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7,api_key=API_KEY,streaming=True)

model_with_tools = model.bind_tools(tool_list)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_with_llm(state:ChatbotState) :
    """LLM node that may answer or request a tool call"""

    messages = state["messages"]
    
    response = model_with_tools.invoke(messages)

    return {"messages":[response]}

tool_node =ToolNode(tool_list)

conn = sqlite3.connect(database="chatbot.db",check_same_thread=False) # check_same_thread false for running multiple thread 

checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatbotState)

graph.add_node("chat_with_llm" , chat_with_llm)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_with_llm")
graph.add_conditional_edges("chat_with_llm",tools_condition)
graph.add_edge("tools","chat_with_llm")
# graph.add_edge("chat_with_llm",END)

chatbot = graph.compile(checkpointer=checkpointer)

all_thread = set()
for checkpoint in checkpointer.list(None):
    # print(checkpoint)
    all_thread.add(checkpoint.config['configurable']['thread_id'])
