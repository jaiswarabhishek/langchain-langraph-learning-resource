from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from states.chatbot_state import ChatbotState
from nodes.chatbot_nodes import make_process_user_message


def build_graph(
    model_name: str,
    temperature: float,
    reasoning_effort: str,
    streaming: bool,
):
    """
    Compiles and returns the LangGraph chatbot with an InMemorySaver checkpointer.
    Cache this in Streamlit with @st.cache_resource so it's built once per
    unique combination of settings.
    """
    checkpointer = InMemorySaver()

    node_fn = make_process_user_message(model_name, temperature, reasoning_effort, streaming)

    graph = StateGraph(ChatbotState)
    graph.add_node("process_user_message", node_fn)
    graph.add_edge(START, "process_user_message")
    graph.add_edge("process_user_message", END)

    return graph.compile(checkpointer=checkpointer)