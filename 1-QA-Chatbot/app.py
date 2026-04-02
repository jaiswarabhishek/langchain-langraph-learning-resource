import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "conversation-qa-chatbot"


# Prompt Template
system_prompt = """You are a helpful assistant for answering questions about a given context.
Use the provided context to answer the question. If you don't know the answer, say you don't know."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question, api_key, llm, temperature, max_tokens):
    try:
        llm = ChatGroq(
            model=llm,
            groq_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer

    except Exception as e:
        raise RuntimeError(f"LLM Error: {str(e)}")


# App Title
st.title("Conversation QA Chatbot 🤖✨")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password")

model_options = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-prompt-guard-2-86m",
    "llama-3.3-70b-versatile"
]
llm = st.sidebar.selectbox("Select LLM Model", model_options)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 500, 50)

# User Input
st.write("### Ask a question about the context:")
user_input = st.text_input("Your Question")


# Action
if st.button("Get Answer"):
    # Validation
    if not api_key:
        st.error("❌ Please enter your Groq API key.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("⏳ Generating response..."):
            try:
                response = generate_response(user_input, api_key, llm, temperature, max_tokens)
                st.success("✅ Response generated successfully!")
                st.subheader("💬 Answer:")
                st.write(response)

            except RuntimeError as e:
                st.error(f"❌ Error: {e}")
            except Exception as e:
                st.error(f"⚠️ Unexpected Error: {str(e)}")

else:
    st.info("💡 Enter a question and click **Get Answer** to begin.")