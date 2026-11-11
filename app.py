
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
# CONFIGURATION
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

        # Support older local databases.
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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background:
            radial-gradient(
                ellipse at top right,
                #f0eaff 0%,
                #faf8ff 45%,
                #fcfaff 100%
            );
        color: #35285d;
    }

    /* Remove Streamlit's black header */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    .block-container {
        max-width: 820px;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f3efff !important;
        border-right: 1px solid #e0d8f5;
    }

    section[data-testid="stSidebar"] * {
        color: #4d3a91;
    }

    .brand-title {
        font-size: 24px;
        font-weight: 800;
        color: #49329c;
        margin-top: 8px;
    }

    .brand-subtitle {
        color: #9587b8;
        font-size: 12px;
        margin-top: 4px;
    }

    .sidebar-heading {
        font-size: 11px;
        letter-spacing: 1.4px;
        font-weight: 800;
        color: #8874b0;
        margin-top: 22px;
        margin-bottom: 12px;
    }

    section[data-testid="stSidebar"] button {
        background: #ffffff;
        color: #5b42b1;
        border: 1px solid #e1d8ff;
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] button:hover {
        background: #ece5ff;
        color: #49329c;
    }

    /* Chat title */
    .chat-title {
        font-size: 28px;
        font-weight: 800;
        color: #49329c;
        margin-bottom: 3px;
    }

    .chat-subtitle {
        color: #9789b6;
        font-size: 12px;
        margin-bottom: 22px;
    }

    /* Welcome */
    .welcome-title {
        color: #49329c;
        font-size: 30px;
        font-weight: 800;
        text-align: center;
        margin-top: 75px;
    }

    .welcome-description {
        color: #9588b2;
        text-align: center;
        font-size: 14px;
        margin-top: 10px;
        margin-bottom: 30px;
    }

    /* Loading indicator */
    .loading-message {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 2px 12px 2px;
        color: #49329c;
        font-size: 14px;
        font-weight: 500;
    }

    .loading-spinner {
        width: 17px;
        height: 17px;
        border: 2px solid #e1d9ff;
        border-top-color: #7154e8;
        border-radius: 50%;
        animation: lalim-spin 0.8s linear infinite;
        flex-shrink: 0;
    }

    @keyframes lalim-spin {
        to {
            transform: rotate(360deg);
        }
    }

    /* Question label */
    .question-label {
        color: #49329c;
        font-size: 14px;
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    /* Question form */
    [data-testid="stForm"] {
        background: #f0ebff;
        border: 1px solid #e0d6ff;
        border-radius: 18px;
        padding: 14px;
    }

    [data-testid="stForm"] input {
        background: #ffffff !important;
        color: #35285d !important;
        border-radius: 10px !important;
    }

    [data-testid="stForm"] button {
        background: #7154e8 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 42px;
    }

    [data-testid="stForm"] button:hover {
        background: #6042d8 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 12px 4px;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #f8f5ff;
        border: 1px solid #dfd6ff;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary {
        color: #49329c !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #aaa0c4;
        font-size: 11px;
        margin-top: 35px;
    }

    footer {
        visibility: hidden;
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
    """
    Display a custom loading message.
    """
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
    """
    Run the existing rag_query() in a background thread.

    This allows Streamlit to update the loading
    message while the backend processes the question.

    Note: The stage change is time-based because
    rag_query() does not expose separate progress events.
    """

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

        # Estimated transition to answer generation.
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
# LOADING AREA
# Must appear ABOVE the question label.
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
