
import os
import json
import uuid
import time
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime

import streamlit as st

from rag_backend import rag_query, FALLBACK


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LalimSalita",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "lalimsalita_chats.db"
LOGO_PATH = BASE_DIR / "lalimsalita_logo.png"

ERROR_MESSAGE = (
    "May problema sa pagkuha ng sagot. "
    "Pakisubukang muli."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lalimsalita.app")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=15,
    )
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                retrieved_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)

        # Compatibility with older databases
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(messages)"
            ).fetchall()
        }

        if "retrieved_json" not in columns:
            connection.execute("""
                ALTER TABLE messages
                ADD COLUMN retrieved_json TEXT DEFAULT '[]'
            """)


def create_conversation():
    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations
            (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                "Bagong usapan",
                now,
                now,
            ),
        )

    return conversation_id


def list_conversations():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, title
            FROM conversations
            ORDER BY updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def load_messages(conversation_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, retrieved_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []

    for row in rows:
        try:
            retrieved = json.loads(
                row["retrieved_json"] or "[]"
            )
        except (ValueError, TypeError):
            retrieved = []

        messages.append({
            "role": row["role"],
            "content": row["content"],
            "retrieved": retrieved,
        })

    return messages


def save_message(
    conversation_id,
    role,
    content,
    retrieved=None,
):
    now = datetime.now().isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages
            (
                conversation_id,
                role,
                content,
                retrieved_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                str(content),
                json.dumps(
                    retrieved or [],
                    ensure_ascii=False,
                ),
                now,
            ),
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, conversation_id),
        )


def update_conversation_title(conversation_id, question):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT title
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if row and row["title"] == "Bagong usapan":
            connection.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE id = ?
                """,
                (question[:55], conversation_id),
            )


# ============================================================
# SESSION STATE
# ============================================================

initialize_database()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = create_conversation()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# COMPLETE CSS - LIGHT THEME AND VISIBILITY FIX
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       APPLICATION BACKGROUND
       -------------------------------------------------------- */

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #FAF8FF !important;
        color: #35285D !important;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at top right,
                #F0EAFF 0%,
                #FAF8FF 45%,
                #FCFAFF 100%
            ) !important;
    }

    /* --------------------------------------------------------
       HIDE STREAMLIT HEADER
       -------------------------------------------------------- */

    header[data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* --------------------------------------------------------
       MAIN CONTAINER
       -------------------------------------------------------- */

    .block-container {
        max-width: 820px;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* --------------------------------------------------------
       DEFAULT TEXT VISIBILITY
       -------------------------------------------------------- */

    .stApp p,
    .stApp li,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #35285D;
    }

    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #F3EFFF !important;
        border-right: 1px solid #E0D8F5;
    }

    section[data-testid="stSidebar"]
    [data-testid="stSidebarContent"] {
        background: #F3EFFF !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #49329C !important;
    }

    .brand-title {
        font-size: 24px;
        font-weight: 800;
        color: #49329C !important;
        margin-top: 8px;
    }

    .brand-subtitle {
        color: #8273A7 !important;
        font-size: 12px;
        margin-top: 4px;
    }

    .sidebar-heading {
        font-size: 11px;
        letter-spacing: 1.4px;
        font-weight: 800;
        color: #8874B0 !important;
        margin-top: 22px;
        margin-bottom: 12px;
    }

    /* Sidebar buttons */

    section[data-testid="stSidebar"]
    .stButton > button {
        background: #FFFFFF !important;
        color: #49329C !important;
        border: 1px solid #DED5FF !important;
        border-radius: 12px !important;
        min-height: 42px;
    }

    section[data-testid="stSidebar"]
    .stButton > button *,
    section[data-testid="stSidebar"]
    .stButton > button p {
        color: #49329C !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"]
    .stButton > button:hover {
        background: #ECE5FF !important;
        border-color: #B7A5F2 !important;
    }

    /* Sidebar dataset information */

    section[data-testid="stSidebar"]
    [data-testid="stAlert"] {
        background: #EDE7FF !important;
        border: 1px solid #DCD2FF !important;
        border-radius: 12px !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stAlert"] p,
    section[data-testid="stSidebar"]
    [data-testid="stAlert"] span {
        color: #49329C !important;
        opacity: 1 !important;
    }

    /* --------------------------------------------------------
       MAIN CHAT TITLE
       -------------------------------------------------------- */

    .chat-title {
        font-size: 28px;
        font-weight: 800;
        color: #49329C !important;
        margin-bottom: 3px;
    }

    .chat-subtitle {
        color: #8273A7 !important;
        font-size: 12px;
        margin-bottom: 22px;
    }

    /* --------------------------------------------------------
       WELCOME SCREEN
       -------------------------------------------------------- */

    .welcome-title {
        color: #49329C !important;
        font-size: 30px;
        font-weight: 800;
        text-align: center;
        margin-top: 75px;
    }

    .welcome-description {
        color: #8273A7 !important;
        text-align: center;
        font-size: 14px;
        margin-top: 10px;
        margin-bottom: 30px;
    }

    /* --------------------------------------------------------
       SUGGESTED QUESTION BUTTONS
       -------------------------------------------------------- */

    [data-testid="stMain"]
    .stButton > button {
        background: #FFFFFF !important;
        color: #6547C3 !important;
        border: 1px solid #DED5FF !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        min-height: 44px;
    }

    [data-testid="stMain"]
    .stButton > button *,
    [data-testid="stMain"]
    .stButton > button p,
    [data-testid="stMain"]
    .stButton > button span {
        color: #6547C3 !important;
        opacity: 1 !important;
    }

    [data-testid="stMain"]
    .stButton > button:hover {
        background: #EEE8FF !important;
        border-color: #A998ED !important;
    }

    /* --------------------------------------------------------
       CHAT MESSAGE VISIBILITY
       -------------------------------------------------------- */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 12px 4px;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #35285D !important;
        opacity: 1 !important;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background: #E99AA6 !important;
        color: #35285D !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: #F1EAFF !important;
        color: #49329C !important;
    }

    /* --------------------------------------------------------
       LOADING INDICATOR
       -------------------------------------------------------- */

    .loading-message {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 2px 12px 2px;
        color: #49329C !important;
        font-size: 14px;
        font-weight: 500;
    }

    .loading-message span {
        color: #49329C !important;
        opacity: 1 !important;
    }

    .loading-spinner {
        width: 17px;
        height: 17px;
        border: 2px solid #E1D9FF;
        border-top-color: #7154E8;
        border-radius: 50%;
        animation: lalim-spin 0.8s linear infinite;
        flex-shrink: 0;
    }

    @keyframes lalim-spin {
        to {
            transform: rotate(360deg);
        }
    }

    /* --------------------------------------------------------
       QUESTION LABEL
       -------------------------------------------------------- */

    .question-label {
        color: #49329C !important;
        font-size: 14px;
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    /* --------------------------------------------------------
       QUESTION FORM
       -------------------------------------------------------- */

    [data-testid="stForm"] {
        background: #F0EBFF !important;
        border: 1px solid #E0D6FF !important;
        border-radius: 18px !important;
        padding: 14px !important;
    }

    [data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        color: #35285D !important;
        -webkit-text-fill-color: #35285D !important;
        caret-color: #49329C !important;
        border-radius: 10px !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #9486B2 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #9486B2 !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #7154E8 !important;
        box-shadow: 0 0 0 1px #7154E8 !important;
    }

    /* Submit button */

    [data-testid="stForm"]
    button[kind="primary"] {
        background: #7154E8 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 42px;
    }

    [data-testid="stForm"]
    button[kind="primary"] *,
    [data-testid="stForm"]
    button[kind="primary"] p,
    [data-testid="stForm"]
    button[kind="primary"] span {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stForm"]
    button[kind="primary"]:hover {
        background: #6042D8 !important;
    }

    /* --------------------------------------------------------
       RETRIEVAL SOURCE EXPANDER
       -------------------------------------------------------- */

    [data-testid="stExpander"] {
        background: #F8F5FF !important;
        border: 1px solid #DED5FF !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #49329C !important;
    }

    /* --------------------------------------------------------
       FOOTER
       -------------------------------------------------------- */

    .app-footer {
        text-align: center;
        color: #8D80AC !important;
        font-size: 11px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY", "").strip()

    if key:
        return key

    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        return str(key).strip()
    except Exception:
        return ""


# ============================================================
# LOADING INDICATOR
# ============================================================

def show_loading(placeholder, message):
    placeholder.markdown(
        f"""
        <div class="loading-message">
            <div class="loading-spinner"></div>
            <span>{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHATBOT RESPONSE
# ============================================================

def generate_chatbot_response(question, loading_placeholder):

    api_key = get_groq_api_key()

    if not api_key:
        logger.error("GROQ_API_KEY is missing.")

        return {
            "answer": ERROR_MESSAGE,
            "retrieved": [],
        }

    result_holder = {}
    error_holder = {}

    def run_backend():
        try:
            result_holder["data"] = rag_query(
                question,
                api_key=api_key,
            )
        except Exception as exc:
            error_holder["error"] = exc

    worker = threading.Thread(
        target=run_backend,
        daemon=True,
    )

    show_loading(
        loading_placeholder,
        "Hinahanap ang kahulugan...",
    )

    start_time = time.monotonic()
    worker.start()

    current_stage = "retrieval"

    while worker.is_alive():
        elapsed = time.monotonic() - start_time

        if elapsed >= 3 and current_stage == "retrieval":
            current_stage = "generation"

            show_loading(
                loading_placeholder,
                "Binubuo ang sagot...",
            )

        worker.join(timeout=0.2)

    loading_placeholder.empty()

    if "error" in error_holder:
        logger.error(
            "LalimSalita request failed: %s",
            error_holder["error"],
        )

        return {
            "answer": ERROR_MESSAGE,
            "retrieved": [],
        }

    result = result_holder.get("data") or {}

    return {
        "answer": result.get("answer", FALLBACK),
        "retrieved": result.get("retrieved", []),
    }


# ============================================================
# DISPLAY CHAT MESSAGE
# ============================================================

def display_message(message):

    role = message["role"]

    avatar = (
        str(LOGO_PATH)
        if role == "assistant" and LOGO_PATH.exists()
        else "📖" if role == "assistant" else "👤"
    )

    with st.chat_message(role, avatar=avatar):

        st.markdown(message["content"])

        retrieved = message.get("retrieved", [])

        if (
            role == "assistant"
            and retrieved
            and message["content"].strip() != FALLBACK
        ):

            with st.expander(
                "Retrieved source chunks",
                expanded=False,
            ):

                for doc in retrieved:

                    st.markdown(
                        f"**Rank {doc.get('rank', '?')} "
                        f"— PDF Page {doc.get('page', '?')}**"
                    )

                    distance = doc.get("distance")

                    if isinstance(distance, (int, float)):
                        st.caption(
                            f"Cosine distance: {distance:.4f}"
                        )

                    st.write(doc.get("text", ""))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=52)
    else:
        st.markdown("📖")

    st.markdown(
        """
        <div class="brand-title">LalimSalita</div>
        <div class="brand-subtitle">
            Kasama mo sa pag-unawa ng Filipino
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">PAG-AARAL</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Bagong usapan",
        use_container_width=True,
        key="new_conversation",
    ):
        st.session_state.conversation_id = create_conversation()
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="sidebar-heading">MGA NAKARAANG USAPAN</div>',
        unsafe_allow_html=True,
    )

    for conversation in list_conversations():

        if st.button(
            conversation["title"],
            key=f"chat_{conversation['id']}",
            use_container_width=True,
        ):
            st.session_state.conversation_id = conversation["id"]
            st.session_state.pending_question = None
            st.rerun()

    st.markdown(
        '<div class="sidebar-heading">MGA SETTING</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Retrieval settings"):

        st.markdown(
            """
            **Chunk size:** 500 characters

            **Chunk overlap:** 100 characters

            **Top-K:** 3

            **Retrieval:** Hybrid exact-match
            and semantic search

            **Embedding model:** Quantized ONNX
            """
        )

    st.markdown(
        '<div class="sidebar-heading">TUNGKOL SA DATASET</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Ang mga sagot ay nakabatay sa "
        "Talasalitaan Aralin 1–29."
    )


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

st.markdown(
    """
    <div class="chat-title">LalimSalita</div>
    <div class="chat-subtitle">
        Filipino Vocabulary Learning Companion
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

conversation_id = st.session_state.conversation_id
messages = load_messages(conversation_id)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not messages:

    st.markdown(
        """
        <div class="welcome-title">
            Ano ang gusto mong malaman?
        </div>

        <div class="welcome-description">
            Magtanong tungkol sa kahulugan at
            konteksto ng mga salitang Filipino.
        </div>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        "Ano ang kahulugan ng hilahil?",
        "Ano ang ibig sabihin ng dalamhati?",
        "Ano ang kahulugan ng nagpipighati?",
    ]

    cols = st.columns(3)

    for col, example in zip(cols, examples):

        with col:

            if st.button(
                example,
                use_container_width=True,
            ):
                st.session_state.pending_question = example
                st.rerun()


# ============================================================
# DISPLAY CONVERSATION
# ============================================================

for message in messages:
    display_message(message)


# ============================================================
# LOADING AREA ABOVE INPUT
# ============================================================

loading_placeholder = st.empty()


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    """
    <div class="question-label">
        Magtanong sa LalimSalita
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form(
    key="question_form",
    clear_on_submit=True,
    enter_to_submit=True,
):

    input_col, button_col = st.columns(
        [5, 1],
        gap="small",
        vertical_alignment="bottom",
    )

    with input_col:

        question_input = st.text_input(
            "Iyong tanong",
            placeholder="I-type ang iyong tanong dito...",
            label_visibility="collapsed",
        )

    with button_col:

        submitted = st.form_submit_button(
            "Ipadala",
            type="primary",
            use_container_width=True,
        )


# ============================================================
# PROCESS QUESTION
# ============================================================

question = None

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

elif submitted:
    question = question_input.strip()


if question and question.strip():

    question = question.strip()

    try:
        update_conversation_title(
            conversation_id,
            question,
        )

        save_message(
            conversation_id,
            "user",
            question,
        )

        result = generate_chatbot_response(
            question,
            loading_placeholder,
        )

        save_message(
            conversation_id,
            "assistant",
            result["answer"],
            result.get("retrieved", []),
        )

    except Exception:
        logger.exception("Unexpected chatbot error.")

        loading_placeholder.empty()

        save_message(
            conversation_id,
            "assistant",
            ERROR_MESSAGE,
        )

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        LalimSalita · Filipino Vocabulary
        Learning Companion
    </div>
    """,
    unsafe_allow_html=True,
)
