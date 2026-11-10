
import os
import json
import uuid
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st

from rag_backend import rag_query, FALLBACK


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "lalimsalita_chats.db"
LOGO_PATH = BASE_DIR / "lalimsalita_logo.png"

ERROR_MESSAGE = (
    "May problema sa pagkuha ng sagot. "
    "Pakisubukang muli."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lalimsalita.app")

st.set_page_config(
    page_title="LalimSalita",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "L",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SQLITE DATABASE
# ============================================================

def get_connection():
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=15,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    return connection


def get_columns(connection, table):
    return {
        row["name"]
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def initialize_database():
    """
    Create tables and migrate older databases.

    Existing conversations and messages are preserved.
    """

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                visitor_id TEXT
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                retrieved_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id)
            )
        """)

        conversation_columns = get_columns(
            connection, "conversations"
        )

        if "visitor_id" not in conversation_columns:
            connection.execute("""
                ALTER TABLE conversations
                ADD COLUMN visitor_id TEXT
            """)

        message_columns = get_columns(
            connection, "messages"
        )

        if "retrieved_json" not in message_columns:
            connection.execute("""
                ALTER TABLE messages
                ADD COLUMN retrieved_json TEXT DEFAULT '[]'
            """)

        if "created_at" not in message_columns:
            connection.execute("""
                ALTER TABLE messages
                ADD COLUMN created_at TEXT
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_conversations_visitor
            ON conversations(visitor_id)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_messages_conversation
            ON messages(conversation_id)
        """)


def create_conversation():
    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations
            (
                id, title, created_at,
                updated_at, visitor_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                "Bagong usapan",
                now,
                now,
                st.session_state.visitor_id,
            ),
        )

    return conversation_id


def list_conversations():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title
            FROM conversations
            WHERE visitor_id = ?
            ORDER BY updated_at DESC
            """,
            (st.session_state.visitor_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def load_messages(conversation_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                m.role,
                m.content,
                m.retrieved_json
            FROM messages AS m
            INNER JOIN conversations AS c
                ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.visitor_id = ?
            ORDER BY m.id ASC
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

            if not isinstance(retrieved, list):
                retrieved = []

        except (json.JSONDecodeError, TypeError):
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

        cursor = connection.execute(
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
            WHERE id = ?
              AND visitor_id = ?
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
                "Conversation not found for this session."
            )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
              AND visitor_id = ?
            """,
            (
                now,
                conversation_id,
                st.session_state.visitor_id,
            ),
        )


def update_conversation_title(
    conversation_id,
    question,
):
    title = question.strip()

    if len(title) > 45:
        title = title[:42] + "..."

    with get_connection() as connection:
        connection.execute(
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
# INITIALIZE SESSION
# ============================================================

initialize_database()

if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = str(uuid.uuid4())

if "conversation_id" not in st.session_state:
    existing = list_conversations()

    if existing:
        st.session_state.conversation_id = (
            existing[0]["id"]
        )
    else:
        st.session_state.conversation_id = (
            create_conversation()
        )

if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* ----------------------------------------------------------
   GLOBAL
---------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(
            circle at 80% 0%,
            rgba(219, 205, 255, 0.30),
            transparent 40%
        ),
        #FCFAFF !important;

    color: #302653 !important;
}

html, body {
    font-family: "Segoe UI", Arial, sans-serif;
}

.block-container {
    max-width: 900px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* Remove dark top header */

header[data-testid="stHeader"] {
    background: #FCFAFF !important;
}

[data-testid="stDecoration"] {
    background: #7154E8 !important;
}

/* ----------------------------------------------------------
   SIDEBAR
---------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #F5F1FF !important;
    border-right: 1px solid #E3DBFF !important;
}

section[data-testid="stSidebar"]
[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}

.sidebar-heading {
    color: #796A9D;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.3px;
    margin-top: 24px;
    margin-bottom: 10px;
}

.brand-icon {
    width: 48px;
    height: 48px;
    background: #7154E8;
    border-radius: 14px;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    font-weight: 800;
}

.brand-title {
    color: #48319B;
    font-size: 23px;
    font-weight: 800;
    margin-top: 10px;
}

.brand-subtitle {
    color: #8275A4;
    font-size: 12px;
    margin-top: 5px;
}

/* ----------------------------------------------------------
   MAIN HEADER
---------------------------------------------------------- */

.chat-title {
    color: #49329C;
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 2px;
}

.chat-subtitle {
    color: #8175A2;
    font-size: 12px;
    margin-bottom: 20px;
}

/* ----------------------------------------------------------
   WELCOME SCREEN
---------------------------------------------------------- */

.welcome-container {
    text-align: center;
    margin-top: 80px;
    margin-bottom: 35px;
}

.welcome-title {
    color: #49329C;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.3;
    margin-bottom: 12px;
}

.welcome-description {
    color: #8175A2;
    font-size: 15px;
    line-height: 1.7;
}

/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

.stButton > button {
    border-radius: 12px !important;
    border: 1px solid #DDD4FF !important;
    background: #FFFFFF !important;
    color: #5A43B3 !important;
    font-weight: 600 !important;

    min-height: 46px !important;
    height: auto !important;

    padding: 10px 14px !important;

    white-space: normal !important;
    overflow-wrap: anywhere !important;
    line-height: 1.4 !important;
}

.stButton > button:hover {
    background: #EEE9FF !important;
    border-color: #A99AF7 !important;
    color: #49329C !important;
}

/* ----------------------------------------------------------
   CHAT MESSAGES
---------------------------------------------------------- */

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding-top: 10px;
    padding-bottom: 10px;
}

[data-testid="stChatMessageContent"] {
    color: #302653 !important;
    font-size: 15px;
    line-height: 1.8;
}

[data-testid="stChatMessageAvatarUser"] {
    background: #E8A0AF !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: #EEE8FF !important;
}

/* ----------------------------------------------------------
   QUESTION FORM
---------------------------------------------------------- */

[data-testid="stForm"] {
    background: #F0EBFF !important;
    border: 1px solid #DED4FF !important;
    border-radius: 20px !important;
    padding: 14px !important;

    box-shadow:
        0 8px 28px rgba(93, 70, 171, 0.07);
}

[data-testid="stForm"] input {
    background: #FFFFFF !important;
    color: #302653 !important;

    border-radius: 12px !important;
    border: 1px solid #DED4FF !important;

    min-height: 46px !important;
}

[data-testid="stForm"] input::placeholder {
    color: #958BAF !important;
    opacity: 1 !important;
}

/* Force purple submit button */

div[data-testid="stForm"]
button[kind="primary"],
div[data-testid="stForm"]
button[data-testid="stBaseButton-primaryFormSubmit"] {
    background-color: #7154E8 !important;
    border-color: #7154E8 !important;
    color: #FFFFFF !important;

    min-height: 46px !important;
    border-radius: 12px !important;
}

div[data-testid="stForm"]
button[kind="primary"]:hover,
div[data-testid="stForm"]
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background-color: #6042D8 !important;
    border-color: #6042D8 !important;
    color: #FFFFFF !important;
}

/* ----------------------------------------------------------
   RETRIEVAL SETTINGS
---------------------------------------------------------- */

section[data-testid="stSidebar"]
[data-testid="stExpander"] {
    background: #F3EEFF !important;
    border: 1px solid #DCD2FF !important;
    border-radius: 12px !important;
}

section[data-testid="stSidebar"]
[data-testid="stExpander"] summary {
    background: #ECE5FF !important;
    color: #38266F !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"]
[data-testid="stExpander"] summary * {
    color: #38266F !important;
    fill: #38266F !important;
}

section[data-testid="stSidebar"]
[data-testid="stExpander"] p,
section[data-testid="stSidebar"]
[data-testid="stExpander"] span,
section[data-testid="stSidebar"]
[data-testid="stExpander"] li {
    color: #38266F !important;
    opacity: 1 !important;
}

/* ----------------------------------------------------------
   SOURCE EXPANDERS
---------------------------------------------------------- */

[data-testid="stExpander"] {
    border: 1px solid #E3DCFF !important;
    border-radius: 12px !important;
    background: #FAF8FF !important;
}

[data-testid="stExpander"] summary {
    color: #49329C !important;
}

/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.app-footer {
    color: #9185B0;
    font-size: 11px;
    text-align: center;
    margin-top: 25px;
    margin-bottom: 12px;
}

footer {
    visibility: hidden;
}

/* ----------------------------------------------------------
   MOBILE RESPONSIVENESS
---------------------------------------------------------- */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .welcome-container {
        margin-top: 45px;
    }

    .welcome-title {
        font-size: 25px;
    }

    .welcome-description {
        font-size: 13px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")

    if key:
        return key.strip()

    try:
        key = st.secrets.get("GROQ_API_KEY")
        return str(key).strip() if key else None

    except (FileNotFoundError, KeyError):
        return None


# ============================================================
# CHATBOT RESPONSE
# ============================================================

def generate_chatbot_response(question):
    try:
        api_key = get_groq_api_key()

        if not api_key:
            logger.error(
                "GROQ_API_KEY is missing"
            )

            return {
                "answer": (
                    "Pansamantalang hindi available "
                    "ang serbisyo ng chatbot. "
                    "Pakisubukang muli."
                ),
                "retrieved": [],
            }

        result = rag_query(
            question,
            api_key=api_key,
        )

        answer = str(
            result.get("answer") or FALLBACK
        ).strip()

        sources = result.get("retrieved") or []

        if not isinstance(sources, list):
            sources = []

        if FALLBACK.casefold() in answer.casefold():
            sources = []

        return {
            "answer": answer,
            "retrieved": sources,
        }

    except Exception:
        logger.exception(
            "LalimSalita request failed"
        )

        return {
            "answer": ERROR_MESSAGE,
            "retrieved": [],
        }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width=55,
        )
    else:
        st.markdown(
            '<div class="brand-icon">L</div>',
            unsafe_allow_html=True,
        )

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
        st.session_state.conversation_id = (
            create_conversation()
        )
        st.rerun()

    st.markdown(
        '<div class="sidebar-heading">MGA NAKARAANG USAPAN</div>',
        unsafe_allow_html=True,
    )

    conversations = list_conversations()

    for conversation in conversations:
        if st.button(
            conversation["title"],
            key=f"conversation_{conversation['id']}",
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

    with st.expander(
        "Retrieval settings",
        expanded=False,
    ):
        st.markdown(
            """
**Chunk size:** 500 characters

**Chunk overlap:** 100 characters

**Top-K:** 3

**Retrieval:** Hybrid exact-match and semantic search

**Embedding model:** Multilingual E5 (quantized ONNX)
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
# MAIN CHAT HEADER
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

    # HTML begins at the left edge to prevent Markdown
    # from interpreting it as a code block.

    st.markdown(
        """
<div class="welcome-container">
    <div class="welcome-title">
        Ano ang gusto mong malaman?
    </div>
    <div class="welcome-description">
        Magtanong tungkol sa kahulugan at konteksto
        ng mga salitang Filipino.
    </div>
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

    for index, example in enumerate(examples):
        with cols[index]:
            if st.button(
                example,
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = example
                st.rerun()


# ============================================================
# DISPLAY CONVERSATION
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

        retrieved = message.get("retrieved", [])

        if (
            role == "assistant"
            and retrieved
            and FALLBACK.casefold()
            not in message["content"].casefold()
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

                    st.caption(
                        "Cosine distance: "
                        f"{float(doc.get('distance') or 0):.4f}"
                    )

                    st.write(
                        doc.get("text", "")
                    )


# ============================================================
# QUESTION INPUT
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
    key="question_form",
    clear_on_submit=True,
    enter_to_submit=True,
):

    input_col, button_col = st.columns(
        [5, 1],
        gap="small",
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

if submitted:
    question = question_input.strip()

elif "pending_question" in st.session_state:
    question = st.session_state.pop(
        "pending_question"
    )

if question and not st.session_state.processing:

    st.session_state.processing = True

    try:
        conversation_id = (
            st.session_state.conversation_id
        )

        update_conversation_title(
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
            result = generate_chatbot_response(
                question
            )

        save_message(
            conversation_id,
            "assistant",
            result["answer"],
            result.get("retrieved", []),
        )

    finally:
        st.session_state.processing = False

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
