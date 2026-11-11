
import os
import json
import uuid
import html
import sqlite3
import logging
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

logger = logging.getLogger("lalimsalita.app")

WELCOME_QUESTIONS = [
    "Ano ang kahulugan ng hilahil?",
    "Ano ang ibig sabihin ng dalamhati?",
    "Ano ang kahulugan ng nagpipighati?",
]


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=20,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                retrieved_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)

        # Support older versions of the messages table.
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(messages)"
            ).fetchall()
        }

        if "retrieved_json" not in columns:
            conn.execute(
                """
                ALTER TABLE messages
                ADD COLUMN retrieved_json TEXT DEFAULT '[]'
                """
            )

        conn.commit()


def create_conversation():
    conversation_id = str(uuid.uuid4())

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations
                (id, title, created_at)
            VALUES (?, ?, ?)
            """,
            (
                conversation_id,
                "Bagong usapan",
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    return conversation_id


def list_conversations():
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, title, created_at
            FROM conversations
            ORDER BY created_at DESC
            """
        ).fetchall()


def load_messages(conversation_id):
    if not conversation_id:
        return []

    with get_connection() as conn:
        rows = conn.execute(
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
        except (TypeError, json.JSONDecodeError):
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
    retrieved_json = json.dumps(
        retrieved or [],
        ensure_ascii=False,
        default=str,
    )

    with get_connection() as conn:
        conn.execute(
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
                retrieved_json,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()


def update_conversation_title(
    conversation_id,
    title,
):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE id = ?
            """,
            (title[:80], conversation_id),
        )
        conn.commit()


def delete_conversation(conversation_id):
    with get_connection() as conn:

        conn.execute(
            """
            DELETE FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )

        conn.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        )

        conn.commit()


# ============================================================
# SESSION STATE
# ============================================================

initialize_database()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "delete_target" not in st.session_state:
    st.session_state.delete_target = None


def start_new_conversation():
    st.session_state.conversation_id = (
        create_conversation()
    )
    st.session_state.pending_question = None
    st.session_state.delete_target = None


def open_conversation(conversation_id):
    st.session_state.conversation_id = conversation_id
    st.session_state.pending_question = None
    st.session_state.delete_target = None


def confirm_delete():
    target = st.session_state.delete_target

    if not target:
        return

    delete_conversation(target)

    if st.session_state.conversation_id == target:
        st.session_state.conversation_id = None

    st.session_state.delete_target = None


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

:root {
    --purple: #6952e8;
    --purple-dark: #4d359f;
    --purple-light: #f0ebff;
    --text: #302653;
    --muted: #8175a2;
    --border: #e2d9ff;
}

.stApp {
    background:
        radial-gradient(
            ellipse at 90% 0%,
            #f1edff 0%,
            transparent 45%
        ),
        #fbfaff;
    color: var(--text);
}

[data-testid="stDecoration"] {
    display: none;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1000px;
    padding-top: 1.6rem;
    padding-bottom: 7rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: #f3efff;
    border-right: 1px solid #e3dcfa;
}

[data-testid="stSidebar"] * {
    color: #46348d;
}

.sidebar-brand {
    padding: 12px 4px 22px;
    border-bottom: 1px solid #dcd4ef;
    margin-bottom: 22px;
}

.sidebar-brand h2 {
    color: #4d359f !important;
    font-size: 25px;
    font-weight: 800;
    margin: 0;
}

.sidebar-brand p {
    color: #8175a2 !important;
    font-size: 12px;
    margin-top: 6px;
}

.sidebar-heading {
    color: #8172a6 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.4px;
    margin: 18px 0 12px;
    text-transform: uppercase;
}

[data-testid="stSidebar"] .stButton > button {
    background: #ffffff !important;
    border: 1px solid #e1d9ff !important;
    border-radius: 12px !important;
    color: #4d359f !important;
    min-height: 42px;
    transition: 0.2s ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #ebe4ff !important;
    border-color: #ad9cff !important;
    color: #4d359f !important;
}

[data-testid="stSidebar"] .stButton > button p {
    color: inherit !important;
}


/* =========================================================
   HEADER
   ========================================================= */

.chat-brand {
    padding: 4px 0 22px;
    border-bottom: 1px solid #ded7ed;
    margin-bottom: 28px;
}

.chat-brand h1 {
    color: #4d359f !important;
    font-size: 29px;
    font-weight: 800;
    margin: 0;
}

.chat-brand p {
    color: #8175a2 !important;
    font-size: 12px;
    margin: 4px 0 0;
}


/* =========================================================
   USER CHAT - RIGHT SIDE
   ========================================================= */

.user-message-row {
    display: flex;
    justify-content: flex-end;
    width: 100%;
    margin: 16px 0 22px;
}

.user-message-bubble {
    display: inline-block;
    width: fit-content;
    max-width: 75%;
    padding: 12px 18px;
    background: #6952e8;
    color: #ffffff !important;
    border-radius: 18px 18px 5px 18px;
    font-size: 15px;
    line-height: 1.65;
    text-align: left;
    overflow-wrap: anywhere;
    box-shadow: 0 3px 12px rgba(105, 82, 232, 0.12);
}

.user-message-bubble,
.user-message-bubble * {
    color: #ffffff !important;
}


/* =========================================================
   ASSISTANT CHAT - LEFT SIDE
   ========================================================= */

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
    margin-bottom: 12px;
}

[data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border: 1px solid #ece6ff;
    border-radius: 5px 18px 18px 18px;
    padding: 14px 18px;
    color: #302653 !important;
    box-shadow: 0 3px 14px rgba(70, 45, 120, 0.04);
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessageContent"] em {
    color: #302653 !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: #f1eaff !important;
    border-radius: 11px !important;
}


/* =========================================================
   RETRIEVED SOURCES
   ========================================================= */

[data-testid="stExpander"] {
    border: 1px solid #ded5f6 !important;
    border-radius: 11px !important;
    background: #faf8ff !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #4d359f !important;
}


/* =========================================================
   WELCOME SCREEN
   ========================================================= */

.welcome {
    text-align: center;
    margin: 65px auto 24px;
}

.welcome h2 {
    color: #4d359f !important;
    font-size: 29px;
    font-weight: 800;
}

.welcome p {
    color: #8175a2 !important;
    font-size: 14px;
}


/* =========================================================
   FIXED BOTTOM CHAT INPUT
   ========================================================= */

[data-testid="stBottom"] {
    background: linear-gradient(
        to bottom,
        transparent,
        #fbfaff 25%
    ) !important;
    padding: 0 !important;
}

[data-testid="stBottom"] > div {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 8px 12px 12px !important;
    background: transparent !important;
}

[data-testid="stChatInput"] {
    min-height: 52px !important;
    background: #f0ebff !important;
    border: 1px solid #ded4ff !important;
    border-radius: 14px !important;
    padding: 4px 6px !important;
    box-shadow: 0 5px 18px rgba(80, 55, 140, 0.06);
}

[data-testid="stChatInput"] textarea {
    min-height: 36px !important;
    max-height: 100px !important;
    padding: 8px 12px !important;
    color: #302653 !important;
    -webkit-text-fill-color: #302653 !important;
    background: #ffffff !important;
    border-radius: 9px !important;
    font-size: 14px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8b7cae !important;
    -webkit-text-fill-color: #8b7cae !important;
}

[data-testid="stChatInput"] button {
    width: 90px !important;
    min-width: 90px !important;
    height: 38px !important;
    background: #6952e8 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    position: relative !important;
}

[data-testid="stChatInput"] button:hover {
    background: #5741cb !important;
}

[data-testid="stChatInput"] button svg {
    display: none !important;
}

[data-testid="stChatInput"] button::after {
    content: "Ipadala";
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 600;
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;
    }

    .chat-brand h1 {
        font-size: 25px;
    }

    .user-message-bubble {
        max-width: 90%;
        font-size: 14px;
        padding: 10px 14px;
    }

    .welcome {
        margin-top: 35px;
    }

    .welcome h2 {
        font-size: 24px;
    }

    [data-testid="stBottom"] > div {
        padding: 6px 8px 10px !important;
    }

    [data-testid="stChatInput"] button {
        width: 76px !important;
        min-width: 76px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>📖 LalimSalita</h2>
            <p>Kasama mo sa pag-unawa ng Filipino</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">PAG-AARAL</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "＋ Bagong usapan",
        use_container_width=True,
        key="new_conversation",
    ):
        start_new_conversation()
        st.rerun()

    st.markdown(
        '<div class="sidebar-heading">'
        'MGA NAKARAANG USAPAN'
        '</div>',
        unsafe_allow_html=True,
    )

    conversations = list_conversations()

    if not conversations:
        st.caption("Wala pang nakaraang usapan.")

    for conversation in conversations:

        conversation_id = conversation["id"]
        title = conversation["title"]

        col_title, col_delete = st.columns(
            [5, 1],
            gap="small",
        )

        with col_title:

            display_title = (
                title[:32] + "..."
                if len(title) > 35
                else title
            )

            if st.button(
                display_title,
                key=f"open_{conversation_id}",
                use_container_width=True,
                help=title,
            ):
                open_conversation(conversation_id)
                st.rerun()

        with col_delete:

            if st.button(
                "⋮",
                key=f"menu_{conversation_id}",
                help="Mga opsyon sa usapan",
                use_container_width=True,
            ):
                if (
                    st.session_state.delete_target
                    == conversation_id
                ):
                    st.session_state.delete_target = None
                else:
                    st.session_state.delete_target = (
                        conversation_id
                    )

                st.rerun()

        if (
            st.session_state.delete_target
            == conversation_id
        ):

            with st.container(border=True):

                st.markdown(
                    "**Burahin ang usapang ito?**"
                )

                st.caption(
                    "Hindi na maibabalik ang mga mensahe."
                )

                yes_col, no_col = st.columns(2)

                with yes_col:

                    if st.button(
                        "Burahin",
                        key=f"confirm_{conversation_id}",
                        use_container_width=True,
                    ):
                        confirm_delete()
                        st.rerun()

                with no_col:

                    if st.button(
                        "Kanselahin",
                        key=f"cancel_{conversation_id}",
                        use_container_width=True,
                    ):
                        st.session_state.delete_target = None
                        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="chat-brand">
        <h1>LalimSalita</h1>
        <p>Filipino Vocabulary Learning Companion</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESPONSE HANDLING
# ============================================================

def normalize_response(result):

    if isinstance(result, str):
        return result, []

    if isinstance(result, dict):

        answer = (
            result.get("answer")
            or result.get("response")
            or result.get("text")
            or FALLBACK
        )

        retrieved = (
            result.get("retrieved")
            or result.get("sources")
            or []
        )

        return str(answer), retrieved

    return FALLBACK, []


def display_sources(retrieved):

    if not retrieved:
        return

    with st.expander(
        "Retrieved source chunks",
        expanded=False,
    ):

        for index, item in enumerate(
            retrieved,
            start=1,
        ):

            if not isinstance(item, dict):
                st.write(str(item))
                continue

            page = item.get("page", "N/A")

            match_type = item.get(
                "match_type",
                "semantic",
            )

            text = item.get("text", "")

            st.markdown(
                f"**Source {index} — Page {page}**"
            )

            st.caption(
                f"Retrieval: {match_type}"
            )

            st.write(text)

            if index < len(retrieved):
                st.divider()


# ============================================================
# CHAT MESSAGE DISPLAY
# ============================================================

def display_message(message):

    role = message["role"]

    if role == "user":

        # Render user messages on the RIGHT.
        # This avoids Streamlit avatar CSS selector issues.

        safe_content = html.escape(
            message["content"]
        ).replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="user-message-row">
                <div class="user-message-bubble">
                    {safe_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        # Assistant messages remain on the LEFT.

        with st.chat_message(
            "assistant",
            avatar="📖",
        ):

            st.markdown(
                message["content"]
            )

            display_sources(
                message.get("retrieved", [])
            )


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_answer(question):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get(
                "GROQ_API_KEY",
                "",
            )
        except Exception:
            api_key = ""

    if not api_key:

        logger.error(
            "GROQ_API_KEY is missing."
        )

        return (
            "Pansamantalang hindi available ang "
            "serbisyo ng chatbot. "
            "Pakisubukang muli.",
            [],
        )

    try:

        result = rag_query(
            question,
            api_key=api_key,
        )

        return normalize_response(result)

    except Exception:

        logger.exception(
            "Chatbot request failed."
        )

        return (
            "May problema sa pagkuha ng sagot. "
            "Pakisubukang muli.",
            [],
        )


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(question):

    question = (question or "").strip()

    if not question:
        return

    conversation_id = (
        st.session_state.conversation_id
    )

    if not conversation_id:

        conversation_id = create_conversation()

        st.session_state.conversation_id = (
            conversation_id
        )

    previous_messages = load_messages(
        conversation_id
    )

    if not previous_messages:

        update_conversation_title(
            conversation_id,
            question,
        )

    # Save user question.
    save_message(
        conversation_id,
        "user",
        question,
    )

    # Display user question immediately.
    display_message({
        "role": "user",
        "content": question,
        "retrieved": [],
    })

    # Generate and display assistant response.
    with st.chat_message(
        "assistant",
        avatar="📖",
    ):

        with st.status(
            "Hinahanap ang kahulugan...",
            expanded=False,
        ) as status:

            status.write(
                "Sinusuri ang mga kaugnay na "
                "salita sa talasalitaan..."
            )

            answer, retrieved = (
                generate_answer(question)
            )

            status.update(
                label="Tapos na ang paghahanap.",
                state="complete",
                expanded=False,
            )

        st.markdown(answer)

        display_sources(retrieved)

    # Save assistant response.
    save_message(
        conversation_id,
        "assistant",
        answer,
        retrieved,
    )

    st.rerun()


# ============================================================
# DISPLAY CONVERSATION
# ============================================================

current_id = st.session_state.conversation_id

messages = load_messages(current_id)

if messages:

    for message in messages:
        display_message(message)

else:

    st.markdown(
        """
        <div class="welcome">
            <h2>Ano ang gusto mong malaman?</h2>
            <p>
                Magtanong tungkol sa kahulugan at
                konteksto ng mga salitang Filipino.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(
        3,
        gap="small",
    )

    for index, suggested in enumerate(
        WELCOME_QUESTIONS
    ):

        with cols[index]:

            label = (
                suggested[:29] + "..."
                if len(suggested) > 32
                else suggested
            )

            if st.button(
                label,
                key=f"suggested_{index}",
                use_container_width=True,
                help=suggested,
            ):

                st.session_state.pending_question = (
                    suggested
                )

                st.rerun()


# ============================================================
# FIXED BOTTOM CHAT INPUT
# ============================================================

question = st.chat_input(
    "I-type ang iyong tanong dito...",
    key="lalimsalita_chat_input",
)

pending = st.session_state.pending_question

if pending:

    st.session_state.pending_question = None

    process_question(pending)

elif question:

    process_question(question)
