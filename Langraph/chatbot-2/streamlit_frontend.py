import streamlit as st
from langgraph_backend import chatbot, all_thread
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid

st.title("LangGraph Chatbot")

# -----------------------------
# Initialize current chat
# -----------------------------
if "current_chat_id" not in st.session_state:

    if all_thread:
        # load first existing thread
        st.session_state.current_chat_id = list(all_thread)[0]
    else:
        # create new
        new_id = str(uuid.uuid4())
        all_thread.add(new_id)
        st.session_state.current_chat_id = new_id


# -----------------------------
# Sidebar (only UUIDs)
# -----------------------------
with st.sidebar:
    st.subheader("📝 New Chat")

    if st.button("➕ Start New Chat"):
        new_id = str(uuid.uuid4())
        all_thread.add(new_id)
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.write("---")
    st.subheader("💬 Chats")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        # Save locally or pass to a processing function
        with open("temp_doc.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Trigger the RAG processing (defined in your backend)
        from langgraph_backend import process_pdf
        process_pdf("temp_doc.pdf")
        st.success("PDF indexed!")

    for tid in all_thread:
        label = tid
        btn_type = "primary" if tid == st.session_state.current_chat_id else "secondary"

        if st.button(label, key=tid, type=btn_type):
            st.session_state.current_chat_id = tid
            st.rerun()


# -----------------------------
# Load chat messages (PERSISTENT)
# -----------------------------
thread_id = st.session_state.current_chat_id
state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
messages = state.values.get("messages", [])


# -----------------------------
# Display messages
# -----------------------------
for msg in messages:
    with st.chat_message("user" if msg.type == "human" else "assistant"):
        st.markdown(msg.content)


# -----------------------------
# Chat input
# -----------------------------
if prompt := st.chat_input("What is up?"):

    st.chat_message("user").markdown(prompt)

    # Stream AI response
    # with st.chat_message("assistant"):
    #     ai_text = st.write_stream(
            
    #         chunk.content
    #         for chunk, metadata in chatbot.stream(
    #             {"messages": [HumanMessage(content=prompt)]},
    #             config={"configurable": {"thread_id": thread_id}},
    #             stream_mode="messages"
    #         ) 
    #         if isinstance(chunk, AIMessage)
    #     )
    with st.chat_message("assistant"):
            # Use a mutable holder so the generator can set/modify it
            status_holder = {"box": None}

            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="messages"
            ):
                    # Lazily create & update the SAME status container when any tool runs
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    # Stream ONLY assistant tokens
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            # Finalize only if a tool was actually used
            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )