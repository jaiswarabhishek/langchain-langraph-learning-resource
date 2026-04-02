import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler

import os

from dotenv import load_dotenv
load_dotenv()

api_wrapper_wikki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper_wikki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="DuckDuckGo Search")


# streamlit

st.title("Search Engine with Agents and Tools 🧑‍💻🔍")
"""
In this example, we're using StreamlitCallbackHandler to stream the agent's thought process in real-time.
This allows us to see the agent's reasoning, the tools it decides to use, and the intermediate steps it takes to arrive at the final answer.
"""


# Sidebar for settings

st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:= st.chat_input("Ask me anything about recent research, Wikipedia topics, or general web search!"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.7, max_tokens=1000)
    tools = [search,wiki_tool,arxiv_tool]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = agent.run(st.session_state.messages, callbacks=[st_callback])
        st.session_state["messages"].append({"role": "assistant", "content": response})

        st.write(response)

    