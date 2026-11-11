
import os
import json
import uuid
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag_backend import rag_query, FALLBACK


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "lalimsalita_chats.db"
LOGO_PATH = BASE_DIR / "lalimsalita_logo.png"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lalimsalita.app")

st.set_page_config(
    page_title="LalimSalita",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "L",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE
# ============================================================

def db_connection():

    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=15,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    with db_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                visitor_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                retrieved_json TEXT DEFAULT '[]',
                created_at TEXT
            )
        """)

        for table, column, definition in [
            ("conversations", "visitor_id", "TEXT"),
            ("messages", "retrieved_json", "TEXT DEFAULT '[]'"),
            ("messages", "created_at", "TEXT"),
        ]:

            columns = {
                row["name"]
                for row in conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }

            if column not in columns:
                conn.execute(
                    f"ALTER TABLE {table} "
                    f"ADD COLUMN {column} {definition}"
                )


def create_conversation():

    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with db_connection() as conn:

        conn.execute(
            """
            INSERT INTO conversations
            (id, title, visitor_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                "Bagong usapan",
                st.session_state.visitor_id,
                now,
                now,
            ),
        )

    return conversation_id


def get_conversations():

    with db_connection() as conn:

        rows = conn.execute(
            """
            SELECT id, title
            FROM conversations
            WHERE visitor_id = ?
            ORDER BY updated_at DESC
            """,
            (st.session_state.visitor_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_messages(conversation_id):

    with db_connection() as conn:

        rows = conn.execute(
            """
            SELECT m.role, m.content, m.retrieved_json
            FROM messages m
            JOIN conversations c
              ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.visitor_id = ?
            ORDER BY m.id
            """,
            (
                conversation_id,
                st.session_state.visitor_id,
            ),
        ).fetchall()

    messages = []

    for row in rows:

        try:
            retrieved = json.loads(
                row["retrieved_json"] or "[]"
            )
        except (ValueError, TypeError):
            retrieved = []

        if not isinstance(retrieved, list):
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

    with db_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO messages
            (
                conversation_id,
                role,
                content,
                retrieved_json,
                created_at
            )
            SELECT id, ?, ?, ?, ?
            FROM conversations
            WHERE id = ? AND visitor_id = ?
            """,
            (
                role,
                content,
                json.dumps(
                    retrieved or [],
                    ensure_ascii=False,
                ),
                now,
                conversation_id,
                st.session_state.visitor_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                "Conversation not found."
            )

        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ? AND visitor_id = ?
            """,
            (
                now,
                conversation_id,
                st.session_state.visitor_id,
            ),
        )


def update_title(conversation_id, question):

    title = question.strip()[:45]

    with db_connection() as conn:

        conn.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE id = ?
              AND visitor_id = ?
              AND title = 'Bagong usapan'
            """,
            (
                title,
                conversation_id,
                st.session_state.visitor_id,
            ),
        )


# ============================================================
# SESSION
# ============================================================

initialize_database()

if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = str(uuid.uuid4())

if "conversation_id" not in st.session_state:

    conversations = get_conversations()

    st.session_state.conversation_id = (
        conversations[0]["id"]
        if conversations
        else create_conversation()
    )


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            #EFE8FF 0%,
            transparent 42%
        ),
        #FCFAFF !important;
    color: #302653;
}

.block-container {
    max-width: 900px !important;
    padding-top: 2rem !important;
}

section[data-testid="stSidebar"] {
    background: #F5F1FF !important;
    border-right: 1px solid #DED6F7;
}

.chat-title {
    color: #49329C;
    font-size: 28px;
    font-weight: 800;
}

.chat-subtitle {
    color: #8175A2;
    font-size: 12px;
    margin-bottom: 22px;
}

.sidebar-title {
    color: #49329C;
    font-size: 23px;
    font-weight: 800;
}

.sidebar-subtitle {
    color: #8175A2;
    font-size: 12px;
}

.sidebar-heading {
    color: #8273A6;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-top: 25px;
    margin-bottom: 10px;
}

.welcome {
    text-align: center;
    margin-top: 75px;
    margin-bottom: 35px;
}

.welcome h1 {
    color: #49329C;
    font-size: 32px;
    font-weight: 800;
}

.welcome p {
    color: #8175A2;
    font-size: 15px;
}

.stButton > button {
    background: #FFFFFF !important;
    color: #5A43B3 !important;
    border: 1px solid #DDD4FF !important;
    border-radius: 12px !important;
    min-height: 44px !important;
    height: auto !important;
    white-space: normal !important;
}

.stButton > button:hover {
    background: #EEE9FF !important;
    color: #49329C !important;
}

[data-testid="stForm"] {
    background: #F0EBFF !important;
    border: 1px solid #DED4FF !important;
    border-radius: 18px !important;
    padding: 14px !important;
}

[data-testid="stForm"] input {
    background: white !important;
    color: #302653 !important;
    border: 1px solid #DDD4FF !important;
    border-radius: 12px !important;
    min-height: 46px !important;
}

[data-testid="stForm"] input::placeholder {
    color: #8D83A8 !important;
    opacity: 1 !important;
}

[data-testid="stForm"] button[kind="primary"],
[data-testid="stForm"] button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: #7154E8 !important;
    color: white !important;
    border: 1px solid #7154E8 !important;
    min-height: 46px !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageContent"] {
    color: #302653 !important;
}

[data-testid="stExpander"] {
    background: #F8F5FF !important;
    border: 1px solid #DDD4FF !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: #49329C !important;
}

.app-footer {
    text-align: center;
    color: #9589B1;
    font-size: 11px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GROQ KEY
# ============================================================

def get_groq_api_key():

    key = os.getenv(
        "GROQ_API_KEY",
        "",
    ).strip()

    if key:
        return key

    try:
        return str(
            st.secrets.get("GROQ_API_KEY", "")
        ).strip()

    except Exception:
        return ""


# ============================================================
# CHATBOT
# ============================================================

def generate_response(question):

    api_key = get_groq_api_key()

    if not api_key:

        logger.error(
            "GROQ_API_KEY is missing."
        )

        return {
            "answer": (
                "Pansamantalang hindi available "
                "ang chatbot. Pakisubukang muli."
            ),
            "retrieved": [],
        }

    try:

        result = rag_query(
            question,
            api_key=api_key,
        )

        answer = str(
            result.get("answer") or FALLBACK
        )

        retrieved = result.get(
            "retrieved",
            [],
        )

        if FALLBACK.casefold() in answer.casefold():
            retrieved = []

        return {
            "answer": answer,
            "retrieved": retrieved,
        }

    except Exception:

        logger.exception(
            "Chatbot request failed."
        )

        return {
            "answer": (
                "May problema sa pagkuha ng sagot. "
                "Pakisubukang muli."
            ),
            "retrieved": [],
        }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=55)

    st.markdown(
        """
<div class="sidebar-title">LalimSalita</div>
<div class="sidebar-subtitle">
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
        key="new_chat",
    ):

        st.session_state.conversation_id = (
            create_conversation()
        )

        st.rerun()

    st.markdown(
        '<div class="sidebar-heading">MGA NAKARAANG USAPAN</div>',
        unsafe_allow_html=True,
    )

    for conversation in get_conversations():

        if st.button(
            conversation["title"],
            key=f"chat_{conversation['id']}",
            use_container_width=True,
        ):

            st.session_state.conversation_id = (
                conversation["id"]
            )

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

**Retrieval:** Hybrid exact-match and semantic search

**Embedding model:** Multilingual E5 (hosted inference)
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
# MAIN HEADER
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

messages = get_messages(conversation_id)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not messages:

    st.markdown(
        """
<div class="welcome">
    <h1>Ano ang gusto mong malaman?</h1>
    <p>
        Magtanong tungkol sa kahulugan at
        konteksto ng mga salitang Filipino.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        "Ano ang kahulugan ng hilahil?",
        "Ano ang ibig sabihin ng dalamhati?",
        "Ano ang kahulugan ng nagpipighati?",
    ]

    columns = st.columns(3)

    for index, example in enumerate(examples):

        with columns[index]:

            if st.button(
                example,
                key=f"example_{index}",
                use_container_width=True,
            ):

                st.session_state.pending_question = example
                st.rerun()


# ============================================================
# DISPLAY MESSAGES
# ============================================================

for message in messages:

    role = message["role"]

    if role not in ("user", "assistant"):
        continue

    avatar = None

    if role == "assistant" and LOGO_PATH.exists():
        avatar = str(LOGO_PATH)

    with st.chat_message(
        role,
        avatar=avatar,
    ):

        st.markdown(message["content"])

        retrieved = message.get("retrieved") or []

        if (
            role == "assistant"
            and retrieved
            and FALLBACK.casefold()
            not in message["content"].casefold()
        ):

            with st.expander(
                "Retrieved source chunks"
            ):

                for item in retrieved:

                    st.markdown(
                        f"**Rank {item.get('rank', '?')} "
                        f"— Page {item.get('page', '?')}**"
                    )

                    st.caption(
                        f"Match: {item.get('match_type', 'semantic')}"
                    )

                    st.write(
                        item.get("text", "")
                    )


# ============================================================
# QUESTION FORM
# ============================================================

st.markdown(
    """
<div style="
    color:#49329C;
    font-size:14px;
    font-weight:800;
    margin-top:25px;
    margin-bottom:10px;
">
Magtanong sa LalimSalita
</div>
    """,
    unsafe_allow_html=True,
)

with st.form(
    "question_form",
    clear_on_submit=True,
    enter_to_submit=True,
):

    input_col, button_col = st.columns(
        [5, 1],
        gap="small",
    )

    with input_col:

        question_input = st.text_input(
            "Tanong",
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

if submitted:
    question = question_input.strip()

elif "pending_question" in st.session_state:
    question = st.session_state.pop(
        "pending_question"
    )


if question:

    conversation_id = st.session_state.conversation_id

    update_title(
        conversation_id,
        question,
    )

    save_message(
        conversation_id,
        "user",
        question,
    )

    with st.spinner(
        "Hinahanap ang kahulugan..."
    ):

        result = generate_response(question)

    save_message(
        conversation_id,
        "assistant",
        result["answer"],
        result.get("retrieved", []),
    )

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="app-footer">
LalimSalita · Filipino Vocabulary Learning Companion
</div>
    """,
    unsafe_allow_html=True,
)
