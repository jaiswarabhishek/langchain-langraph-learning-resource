from states.chatbot_state import ChatbotState
from model.llm import get_llm


def make_process_user_message(
    model_name: str,
    temperature: float,
    reasoning_effort: str,
    streaming: bool,
):
    """
    Factory that creates a LangGraph-compatible node function.
    The LLM is instantiated once and captured in the closure.
    """
    llm = get_llm(model_name, temperature, reasoning_effort, streaming)

    def process_user_message(state: ChatbotState) -> ChatbotState:
        llm_response = llm.invoke(state["conversation"])
        return {"conversation": [llm_response]}

    return process_user_message