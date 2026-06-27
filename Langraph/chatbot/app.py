"""
app.py  ──  LangGraph Chatbot  ·  Full Streamlit UI
─────────────────────────────────────────────────────
Run:  streamlit run app.py
"""

import uuid
import json
import datetime
import streamlit as st
from langchain_core.messages import HumanMessage

from graph.chatbot_graph import build_graph

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Variables ── */
:root {
    --bg:        #080a0f;
    --surface:   #0e1117;
    --panel:     #13161e;
    --border:    #1e2330;
    --border2:   #252d40;
    --accent:    #4f8ef7;
    --accent2:   #7c6af7;
    --acsoft:    rgba(79,142,247,0.10);
    --acsoft2:   rgba(124,106,247,0.10);
    --green:     #3ecf8e;
    --red:       #f76f6f;
    --text:      #dde1f0;
    --muted:     #4a5068;
    --muted2:    #6b728f;
    --radius:    12px;
    --font:      'IBM Plex Sans', sans-serif;
    --mono:      'IBM Plex Mono', monospace;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main { 
    background: var(--bg) !important; 
    font-family: var(--font);
    color: var(--text);
}
[data-testid="stHeader"]        { background: transparent !important; }
[data-testid="stDecoration"]    { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── Sidebar logo ── */
.sb-logo {
    font-family: var(--mono);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    padding: 4px 0 2px;
    letter-spacing: -0.3px;
}
.sb-logo span { color: var(--accent); }
.sb-tagline {
    font-size: 0.72rem;
    color: var(--muted2);
    margin-bottom: 16px;
}

/* ── Section labels ── */
.sec-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 14px 0 6px;
}

/* ── Chat session cards ── */
.chat-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 12px;
    border-radius: 9px;
    border: 1px solid var(--border);
    margin-bottom: 5px;
    cursor: pointer;
    transition: all 0.15s ease;
    background: transparent;
}
.chat-card:hover  { border-color: var(--border2); background: var(--panel); }
.chat-card.active { border-color: var(--accent);  background: var(--acsoft); }
.chat-card-icon { font-size: 1rem; flex-shrink: 0; }
.chat-card-body { flex: 1; min-width: 0; }
.chat-card-title {
    font-size: 0.82rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text);
}
.chat-card.active .chat-card-title { color: var(--accent); }
.chat-card-meta {
    font-size: 0.68rem;
    color: var(--muted2);
    margin-top: 1px;
}

/* ── Thread badge ── */
.thread-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--acsoft2);
    border: 1px solid var(--accent2);
    border-radius: 20px;
    padding: 2px 10px;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent2);
    margin: 6px 0 14px;
}

/* ── Chat header ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.chat-header-title {
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    flex: 1;
}
.chat-header-badge {
    font-size: 0.7rem;
    color: var(--muted2);
    font-family: var(--mono);
}

/* ── Message bubbles ── */
.msg-container { display: flex; flex-direction: column; gap: 18px; padding-bottom: 12px; }

.msg-row { display: flex; gap: 12px; align-items: flex-start; animation: fadeUp 0.25s ease; }
.msg-row.user-row { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}

.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-family: var(--mono);
    font-weight: 600; flex-shrink: 0;
}
.avatar.ai-av   { background: var(--acsoft);  border: 1px solid var(--accent);  color: var(--accent); }
.avatar.usr-av  { background: var(--acsoft2); border: 1px solid var(--accent2); color: var(--accent2); }

.bubble {
    max-width: 72%;
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 0.88rem;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble.ai-bubble  {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top-left-radius: 3px;
    color: var(--text);
}
.bubble.usr-bubble {
    background: #151a2e;
    border: 1px solid var(--border2);
    border-top-right-radius: 3px;
    color: var(--text);
}

.msg-time {
    font-size: 0.62rem;
    color: var(--muted);
    margin-top: 4px;
    font-family: var(--mono);
}
.user-row .msg-time { text-align: right; }

/* ── Empty state ── */
.empty-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px;
    text-align: center;
}
.empty-glyph { font-size: 3rem; margin-bottom: 14px; opacity: 0.4; }
.empty-title { font-family: var(--mono); font-size: 0.95rem; color: var(--muted2); }
.empty-hint  { font-size: 0.78rem; color: var(--muted); margin-top: 6px; }

/* ── Input row ── */
.stTextInput > div > div > input {
    background: var(--panel) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
    padding: 10px 14px !important;
    caret-color: var(--accent);
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--acsoft) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: var(--font) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: opacity 0.15s !important;
    padding: 8px 18px !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.82 !important; }

/* secondary-style buttons rendered as small sidebar icons */
button[kind="secondary"], .sec-btn button {
    background: var(--panel) !important;
    color: var(--muted2) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.75rem !important;
    padding: 4px 8px !important;
}

/* ── Selectbox / slider ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.82rem !important;
}
.stSlider [data-testid="stSlider"] { accent-color: var(--accent); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 10px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── Rename input ── */
[data-testid="stTextInput"].rename-input > div > div > input {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return str(uuid.uuid4())[:10]


def now_label() -> str:
    return datetime.datetime.now().strftime("%b %d, %H:%M")


def auto_title(messages: list) -> str:
    for m in messages:
        if m["role"] == "user":
            words = m["content"].split()
            title = " ".join(words[:6])
            return title + ("…" if len(words) > 6 else "")
    return "New Chat"


def msg_time(msg: dict) -> str:
    return msg.get("time", "")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def boot():
    if "sessions" not in st.session_state:
        # sessions: {id: {label, created, messages: [{role,content,time}]}}
        st.session_state.sessions = {}

    if "active" not in st.session_state:
        _create_session()

    if "model_name" not in st.session_state:
        st.session_state.model_name = "openai/gpt-oss-120b"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "reasoning_effort" not in st.session_state:
        st.session_state.reasoning_effort = "medium"

    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    if "renaming" not in st.session_state:
        st.session_state.renaming = None   # session id being renamed


def _create_session(label: str = "New Chat") -> str:
    sid = new_session_id()
    st.session_state.sessions[sid] = {
        "label":    label,
        "created":  now_label(),
        "messages": [],
    }
    st.session_state.active = sid
    return sid


boot()


# ─────────────────────────────────────────────────────────────────────────────
# CACHED GRAPH  (rebuilds only when model settings change)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_graph(model_name: str, temperature: float, reasoning_effort: str, streaming: bool):
    return build_graph(model_name, temperature, reasoning_effort, streaming)


def invoke(user_text: str, thread_id: str) -> str:
    graph = get_graph(
        st.session_state.model_name,
        st.session_state.temperature,
        st.session_state.reasoning_effort,
        st.session_state.streaming,
    )
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"conversation": [HumanMessage(content=user_text)]},
        config=config,
    )
    last = result["conversation"][-1]
    return last.content if hasattr(last, "content") else str(last)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo
    st.markdown('<div class="sb-logo">lang<span>graph</span>.chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tagline">persistent multi-session chatbot</div>', unsafe_allow_html=True)

    # New chat button
    if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
        _create_session()
        st.session_state.renaming = None
        st.rerun()

    st.markdown("---")

    # ── Model settings ────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Model Settings</div>', unsafe_allow_html=True)

    model_options = [
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
        "openai/gpt-oss-safeguard-20b",
        "openai/gpt-oss-20b",
    ]
    prev_model = st.session_state.model_name
    st.session_state.model_name = st.selectbox(
        "Model", model_options,
        index=model_options.index(st.session_state.model_name),
        label_visibility="collapsed",
        key="model_select",
    )

    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        key="temp_slider",
    )

    reasoning_options = ["low", "medium", "high", "none"]
    st.session_state.reasoning_effort = st.selectbox(
        "Reasoning Effort", reasoning_options,
        index=reasoning_options.index(st.session_state.reasoning_effort),
        key="reasoning_select",
    )

    st.session_state.streaming = st.toggle(
        "Streaming", value=st.session_state.streaming, key="stream_toggle"
    )

    # warn if model changed mid-chat
    if prev_model != st.session_state.model_name:
        st.caption("⚠️ Model changed — graph will rebuild on next message.")

    st.markdown("---")

    # ── Chat history list ─────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Chat History</div>', unsafe_allow_html=True)

    sessions = st.session_state.sessions
    # newest first
    sorted_ids = sorted(sessions.keys(),
                        key=lambda s: sessions[s]["created"], reverse=True)

    for sid in sorted_ids:
        data     = sessions[sid]
        is_active = sid == st.session_state.active
        n_msgs   = len(data["messages"])
        label    = data["label"]
        created  = data["created"]

        card_class = "chat-card active" if is_active else "chat-card"
        icon = "💬" if is_active else ("🗒️" if n_msgs else "○")

        # Render rename input inline if this session is being renamed
        if st.session_state.renaming == sid:
            new_name = st.text_input(
                "Rename", value=label,
                key=f"rename_input_{sid}",
                label_visibility="collapsed",
            )
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("✓ Save", key=f"save_rename_{sid}", use_container_width=True):
                    st.session_state.sessions[sid]["label"] = new_name.strip() or label
                    st.session_state.renaming = None
                    st.rerun()
            with col_cancel:
                if st.button("✕", key=f"cancel_rename_{sid}", use_container_width=True):
                    st.session_state.renaming = None
                    st.rerun()
            continue   # skip the card row while renaming

        # Card click + action buttons
        col_card, col_edit, col_del = st.columns([7, 1, 1])

        with col_card:
            st.markdown(f"""
            <div class="{card_class}">
                <span class="chat-card-icon">{icon}</span>
                <div class="chat-card-body">
                    <div class="chat-card-title">{label}</div>
                    <div class="chat-card-meta">{created} · {n_msgs} msg{"s" if n_msgs != 1 else ""}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # invisible button that overlays the card for click handling
            if st.button("", key=f"open_{sid}", use_container_width=True,
                         help=f"Open {label}"):
                st.session_state.active = sid
                st.session_state.renaming = None
                st.rerun()

        with col_edit:
            if st.button("✏️", key=f"edit_{sid}", help="Rename"):
                st.session_state.renaming = sid
                st.rerun()

        with col_del:
            if st.button("🗑", key=f"del_{sid}", help="Delete"):
                del st.session_state.sessions[sid]
                if st.session_state.active == sid:
                    remaining = [k for k in st.session_state.sessions]
                    if remaining:
                        st.session_state.active = remaining[-1]
                    else:
                        _create_session()
                st.rerun()

    st.markdown("---")

    # Active thread badge
    st.markdown(
        f'<div class="thread-badge">🔗 thread: {st.session_state.active}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

active_session = st.session_state.sessions[st.session_state.active]
messages       = active_session["messages"]

# ── Chat header ──────────────────────────────────────────────────────────────
col_title, col_info = st.columns([6, 2])
with col_title:
    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-title">💬 {active_session['label']}</div>
    </div>
    """, unsafe_allow_html=True)
with col_info:
    st.markdown(
        f'<div style="text-align:right;font-family:var(--mono);font-size:0.68rem;'
        f'color:var(--muted2);padding-top:4px">'
        f'📦 {st.session_state.model_name.split("-")[0].upper()} · '
        f'🌡 {st.session_state.temperature} · '
        f'🧠 {st.session_state.reasoning_effort}</div>',
        unsafe_allow_html=True,
    )

# ── Message list ─────────────────────────────────────────────────────────────
if not messages:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-glyph">◈</div>
        <div class="empty-title">No messages yet</div>
        <div class="empty-hint">Type a message below to start the conversation.<br>
        Your chat history persists across sessions.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    parts = ['<div class="msg-container">']
    for msg in messages:
        role    = msg["role"]
        content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        time    = msg.get("time", "")

        if role == "user":
            parts.append(f"""
            <div class="msg-row user-row">
                <div class="avatar usr-av">U</div>
                <div>
                    <div class="bubble usr-bubble">{content}</div>
                    <div class="msg-time">{time}</div>
                </div>
            </div>""")
        else:
            parts.append(f"""
            <div class="msg-row">
                <div class="avatar ai-av">AI</div>
                <div>
                    <div class="bubble ai-bubble">{content}</div>
                    <div class="msg-time">{time}</div>
                </div>
            </div>""")

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
col_in, col_btn = st.columns([8, 1])

with col_in:
    user_text = st.text_input(
        "msg",
        placeholder="Send a message…  (press Send or hit Enter)",
        label_visibility="collapsed",
        key="user_input",
    )

with col_btn:
    send = st.button("Send ➤", use_container_width=True, key="send_btn")

# ── Handle send ───────────────────────────────────────────────────────────────
if send and user_text.strip():
    sid  = st.session_state.active
    text = user_text.strip()
    t    = now_label()

    # Append user message
    messages.append({"role": "user", "content": text, "time": t})

    # Auto-title from first user message
    if len(messages) == 1:
        st.session_state.sessions[sid]["label"] = auto_title(messages)

    # Call LangGraph
    with st.spinner("Thinking…"):
        try:
            reply = invoke(text, sid)
        except Exception as e:
            reply = f"⚠️ Error: {e}"

    messages.append({"role": "assistant", "content": reply, "time": now_label()})
    st.rerun()