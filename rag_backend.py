
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

APP_NAME = "LalimSalita"

ERROR_MESSAGE = (
    "May problema sa pagkuha ng sagot. "
    "Pakisubukang muli."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lalimsalita.app")

st.set_page_config(
    page_title="LalimSalita",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id)
            )
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
        rows = connection.execute(
            """
            SELECT id, title
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()

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
        except json.JSONDecodeError:
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
                content,
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
            AND title = 'Bagong usapan'
            """,
            (title, conversation_id),
        )


# ============================================================
# SESSION STATE
# ============================================================

initialize_database()

if "conversation_id" not in st.session_state:
    conversations = list_conversations()

    if conversations:
        st.session_state.conversation_id = (
            conversations[0]["id"]
        )
    else:
        st.session_state.conversation_id = (
            create_conversation()
        )

if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Global background */

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(222, 204, 255, 0.28),
            transparent 35%
        ),
        #FCFAFF;
    color: #302653;
}

/* Main content width */

.block-container {
    max-width: 900px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* Typography */

html, body, p, span, label {
    font-family:
        "Segoe UI",
        Arial,
        sans-serif;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #F5F1FF !important;
    border-right: 1px solid #E3DBFF;
}

section[data-testid="stSidebar"]
[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}

/* Sidebar headings */

.sidebar-heading {
    color: #8A7DAF;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.3px;
    margin-top: 24px;
    margin-bottom: 10px;
}

/* Brand */

.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    background: #7154E8;
    border-radius: 13px;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 800;
}

.brand-title {
    color: #48319B;
    font-size: 23px;
    font-weight: 800;
}

.brand-subtitle {
    color: #9489B5;
    font-size: 12px;
    margin-top: 4px;
}

/* Chat header */

.chat-title {
    color: #49329C;
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 2px;
}

.chat-subtitle {
    color: #9389AF;
    font-size: 12px;
    margin-bottom: 20px;
}

/* Chat messages */

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

/* Buttons */

.stButton > button {
    border-radius: 12px !important;
    border: 1px solid #DDD4FF !important;
    background: #FFFFFF !important;
    color: #5A43B3 !important;
    font-weight: 600 !important;
    min-height: 42px;
}

.stButton > button:hover {
    background: #EEE9FF !important;
    border-color: #A99AF7 !important;
    color: #49329C !important;
}

/* Input form */

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

[data-testid="stForm"] button[kind="primary"] {
    background: #7154E8 !important;
    color: #FFFFFF !important;
    border: none !important;
    min-height: 46px !important;
}

[data-testid="stForm"] button[kind="primary"]:hover {
    background: #6042D8 !important;
    color: #FFFFFF !important;
}

/* Retrieval settings visibility */

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

/* Source expanders */

[data-testid="stExpander"] {
    border: 1px solid #E3DCFF !important;
    border-radius: 12px !important;
    background: #FAF8FF !important;
}

[data-testid="stExpander"] summary {
    color: #49329C !important;
}

/* Footer */

.app-footer {
    color: #A49BBE;
    font-size: 11px;
    text-align: center;
    margin-top: 25px;
    margin-bottom: 12px;
}

/* Hide default Streamlit footer */

footer {
    visibility: hidden;
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
                "GROQ_API_KEY is missing on Render"
            )

            return {
                "answer": (
                    "Hindi pansamantalang available "
                    "ang serbisyo ng chatbot. "
                    "Pakisubukang muli."
                ),
                "retrieved": [],
            }

        result = rag_query(
            question,
            api_key=api_key,
        )

        return {
            "answer": result["answer"],
            "retrieved": result.get(
                "retrieved",
                [],
            ),
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
            width=52,
        )
    else:
        st.markdown(
            '<div class="brand-icon">L</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="brand-title">
            LalimSalita
        </div>
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

        title = conversation["title"]

        if st.button(
            title,
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

            **Retrieval:** Hybrid exact-match
            and semantic search

            **Embedding model:** Multilingual E5
            (quantized ONNX)
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
    <div class="chat-title">
        LalimSalita
    </div>
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
# WELCOME MESSAGE
# ============================================================

if not messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:90px;
            margin-bottom:35px;
        ">
            <div style="
                color:#49329C;
                font-size:32px;
                font-weight:800;
                margin-bottom:12px;
            ">
                Ano ang gusto mong malaman?
            </div>

            <div style="
                color:#9389AF;
                font-size:15px;
            ">
                Magtanong tungkol sa kahulugan at
                konteksto ng mga salitang Filipino.
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
            and message["content"] != FALLBACK
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
                        f"Cosine distance: "
                        f"{doc.get('distance', 0):.4f}"
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
            placeholder=(
                "I-type ang iyong tanong dito..."
            ),
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

        answer = result["answer"]
        retrieved = result.get(
            "retrieved",
            [],
        )

        save_message(
            conversation_id,
            "assistant",
            answer,
            retrieved,
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
        LalimSalita · Filipino Vocabulary
        Learning Companion
    </div>
    """,
    unsafe_allow_html=True,
)
