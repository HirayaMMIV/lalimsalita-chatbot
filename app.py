
import base64
import html
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from rag_backend import rag_query, FALLBACK_ANSWER


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LalimSalita",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "lalimsalita_logo.png"
DB_PATH = BASE_DIR / "lalimsalita_chats.db"

SUGGESTIONS = [
    "Ano ang kahulugan ng hilahil?",
    "Ano ang ibig sabihin ng dalamhati?",
    "Ano ang kahulugan ng nagpipighati?",
]


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def connect_db():
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with connect_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                visitor_id TEXT NOT NULL,
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
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
            )
        """)


def now():
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


def create_conversation(visitor_id, question):
    conversation_id = str(uuid.uuid4())
    timestamp = now()

    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO conversations
            (id, visitor_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                visitor_id,
                question[:50],
                timestamp,
                timestamp,
            ),
        )

    return conversation_id


def get_conversations(visitor_id):
    with connect_db() as connection:
        return connection.execute(
            """
            SELECT id, title
            FROM conversations
            WHERE visitor_id = ?
            ORDER BY updated_at DESC
            """,
            (visitor_id,),
        ).fetchall()


def owns_conversation(conversation_id, visitor_id):
    with connect_db() as connection:
        result = connection.execute(
            """
            SELECT 1 FROM conversations
            WHERE id = ? AND visitor_id = ?
            """,
            (conversation_id, visitor_id),
        ).fetchone()

    return result is not None


def save_message(conversation_id, visitor_id, role, content):
    if not owns_conversation(conversation_id, visitor_id):
        raise PermissionError("Conversation access denied.")

    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO messages
            (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                json.dumps(content, ensure_ascii=False),
                now(),
            ),
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (now(), conversation_id),
        )


def get_messages(conversation_id, visitor_id):
    if not conversation_id:
        return []

    if not owns_conversation(conversation_id, visitor_id):
        return []

    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []

    for row in rows:
        try:
            content = json.loads(row["content"])
        except json.JSONDecodeError:
            content = row["content"]

        messages.append({
            "role": row["role"],
            "content": content,
        })

    return messages


def delete_conversation(conversation_id, visitor_id):
    if not owns_conversation(conversation_id, visitor_id):
        return

    with connect_db() as connection:
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ? AND visitor_id = ?
            """,
            (conversation_id, visitor_id),
        )


initialize_database()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

visitor_param = st.query_params.get("visitor")

if "visitor_id" not in st.session_state:
    try:
        st.session_state.visitor_id = str(
            uuid.UUID(str(visitor_param))
        )
    except (ValueError, TypeError, AttributeError):
        st.session_state.visitor_id = str(uuid.uuid4())

if visitor_param != st.session_state.visitor_id:
    st.query_params["visitor"] = st.session_state.visitor_id

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# LOGO
# --------------------------------------------------

def logo_html(css_class):
    if LOGO_PATH.is_file():
        encoded = base64.b64encode(
            LOGO_PATH.read_bytes()
        ).decode("utf-8")

        return (
            f'<img class="{css_class}" '
            f'src="data:image/png;base64,{encoded}" '
            f'alt="LalimSalita logo">'
        )

    return f'<div class="{css_class} logo-fallback">L</div>'


def assistant_avatar():
    if LOGO_PATH.is_file():
        return str(LOGO_PATH)
    return None


# --------------------------------------------------
# DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap'
);

:root {
    color-scheme: light;
}

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 75% 0%,
            #f2eaff 0%,
            transparent 38%
        ),
        #fcfbff;
    color: #303452;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #f7f4ff;
    border-right: 1px solid #e7dff8;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 15px 0 8px;
}

.brand-logo {
    width: 49px;
    height: 49px;
    object-fit: contain;
}

.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: #4b359e;
}

.brand-caption {
    color: #8881a5;
    font-size: 12px;
    margin-bottom: 28px;
}

.sidebar-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #8176a4;
    margin: 22px 0 10px;
}

/* WELCOME */

.welcome {
    text-align: center;
    margin-top: 10vh;
    margin-bottom: 30px;
}

.welcome-logo {
    width: 90px;
    height: 90px;
    object-fit: contain;
    margin: 0 auto 18px;
}

.logo-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #7257e9;
    color: white;
    border-radius: 20px;
    font-size: 32px;
    font-weight: 800;
}

.welcome-title {
    font-family: 'Outfit', sans-serif;
    color: #42348e;
    font-size: clamp(27px, 4vw, 38px);
    font-weight: 800;
    margin-bottom: 10px;
}

.welcome-description {
    font-size: 15px;
    color: #777b9a;
    line-height: 1.7;
}

/* CHAT HEADER */

.chat-header {
    border-bottom: 1px solid #e9e3f8;
    padding: 8px 0 17px;
    margin-bottom: 22px;
}

.chat-header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 23px;
    font-weight: 800;
    color: #4b359e;
}

.chat-header-subtitle {
    font-size: 12px;
    color: #8b85a4;
}

/* BUTTONS */

.stButton > button {
    background: white !important;
    border: 1px solid #ded5fa !important;
    color: #5a45b4 !important;
    border-radius: 13px !important;
    font-weight: 700 !important;
    min-height: 44px;
    white-space: normal;
}

.stButton > button:hover {
    background: #eee8ff !important;
    border-color: #b4a2f5 !important;
}

[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    text-align: left;
}

/* MESSAGES */

[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 10px 0;
    margin-bottom: 8px;
}

[data-testid="stChatMessage"] p {
    color: #35405d;
    line-height: 1.7;
}

/* QUESTION INPUT */

.input-heading {
    font-family: 'Outfit', sans-serif;
    color: #50429b;
    font-size: 14px;
    font-weight: 700;
    margin: 22px 0 10px;
}

[data-testid="stForm"] {
    background: #f1edff !important;
    border: 1px solid #d9cef8 !important;
    border-radius: 19px !important;
    padding: 12px 14px !important;
    box-shadow: 0 8px 24px #7561be12;
}

[data-testid="stTextInput"] > div > div {
    background: white !important;
    border: 1px solid #d8cefa !important;
    border-radius: 12px !important;
}

[data-testid="stTextInput"] input {
    background: white !important;
    color: #293450 !important;
    -webkit-text-fill-color: #293450 !important;
    font-size: 14px !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #9590ab !important;
}

/* Hide hint, keep Enter submission */

[data-testid="InputInstructions"],
[data-testid="stInputInstructions"] {
    display: none !important;
}

/* SEND BUTTON */

[data-testid="stFormSubmitButton"] button {
    background: #7155e8 !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    min-height: 43px !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #6043d4 !important;
    color: white !important;
}

/* SOURCES */

[data-testid="stExpander"] {
    background: #faf9ff;
    border: 1px solid #e8e1fa;
    border-radius: 13px;
}

.footer {
    text-align: center;
    color: #a09bb6;
    font-size: 12px;
    padding: 25px 0 10px;
}

@media (max-width: 700px) {
    .welcome {
        margin-top: 5vh;
    }

    .block-container {
        padding-top: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# REAL RAG BACKEND CONNECTION
# --------------------------------------------------

def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def answer_from_backend(question):
    api_key = get_groq_api_key()

    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Configure it in Render "
            "Environment or Streamlit secrets."
        )

    result = rag_query(question, api_key)

    answer = str(
        result.get("answer") or FALLBACK_ANSWER
    ).strip()

    retrieved = result.get("retrieved") or []

    unsupported = (
        answer.casefold() == FALLBACK_ANSWER.casefold()
    )

    return {
        "answer": answer,
        "sources": [] if unsupported else retrieved,
        "has_evidence": bool(retrieved) and not unsupported,
    }


# --------------------------------------------------
# DISPLAY RESPONSE
# --------------------------------------------------

def display_response(result):
    if not isinstance(result, dict):
        st.markdown(str(result))
        return

    answer = str(result.get("answer") or "").strip()
    sources = result.get("sources") or []

    unsupported = (
        answer.casefold() == FALLBACK_ANSWER.casefold()
    )

    if answer:
        st.markdown(answer)

    # Hide sources for fallback answers.
    if unsupported or not result.get("has_evidence"):
        return

    if not sources:
        return

    with st.expander(
        f"Tingnan ang mga sanggunian ({len(sources)})"
    ):
        for index, source in enumerate(sources, start=1):
            page = source.get("page", "—")
            source_text = source.get("text", "")
            distance = source.get("distance")
            method = source.get("match_type", "semantic")

            st.markdown(
                f"**Sanggunian {index} · Pahina {page}**"
            )

            st.caption(f"Paraan ng retrieval: {method}")

            if isinstance(distance, (int, float)):
                st.caption(
                    f"Cosine distance: {distance:.4f}"
                )

            st.code(str(source_text), language=None)

            if index < len(sources):
                st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
            {logo_html("brand-logo")}
            <div class="brand-name">LalimSalita</div>
        </div>
        <div class="brand-caption">
            Kasama mo sa pag-unawa ng Filipino
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Bagong usapan",
        use_container_width=True,
    ):
        st.session_state.active_chat = None
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="sidebar-label">MGA NAKARAANG USAPAN</div>',
        unsafe_allow_html=True,
    )

    conversations = get_conversations(
        st.session_state.visitor_id
    )

    if not conversations:
        st.caption("Wala pang nakaraang usapan.")

    for conversation in conversations:
        chat_col, delete_col = st.columns(
            [5, 1],
            vertical_alignment="center",
        )

        with chat_col:
            if st.button(
                conversation["title"],
                key=f"open_{conversation['id']}",
                use_container_width=True,
            ):
                st.session_state.active_chat = conversation["id"]
                st.session_state.pending_question = None
                st.rerun()

        with delete_col:
            if st.button(
                "×",
                key=f"delete_{conversation['id']}",
                help="Burahin ang usapan",
            ):
                delete_conversation(
                    conversation["id"],
                    st.session_state.visitor_id,
                )

                if (
                    st.session_state.active_chat
                    == conversation["id"]
                ):
                    st.session_state.active_chat = None

                st.rerun()

    st.markdown(
        '<div class="sidebar-label">MGA SETTING</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Retrieval settings"):
        st.write("Chunk size: **500** characters")
        st.write("Chunk overlap: **100** characters")
        st.write("Top-K: **3**")
        st.caption("Hybrid exact-match at semantic retrieval")

    st.caption("Talasalitaan Aralin 1–29")


# --------------------------------------------------
# LOAD CONVERSATION
# --------------------------------------------------

messages = get_messages(
    st.session_state.active_chat,
    st.session_state.visitor_id,
)


# --------------------------------------------------
# WELCOME SCREEN
# --------------------------------------------------

if not messages:
    st.markdown(
        f"""
        <div class="welcome">
            {logo_html("welcome-logo")}
            <div class="welcome-title">
                Kumusta! Ako si LalimSalita.
            </div>
            <div class="welcome-description">
                Anong salitang Filipino ang gusto mong
                mas maintindihan ngayon?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestion_columns = st.columns(3)

    for column, suggestion in zip(
        suggestion_columns,
        SUGGESTIONS,
    ):
        with column:
            if st.button(
                suggestion,
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion
                st.rerun()

else:
    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-header-title">
                LalimSalita
            </div>
            <div class="chat-header-subtitle">
                Filipino Vocabulary Learning Companion
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in messages:
        role = message["role"]

        avatar = (
            assistant_avatar()
            if role == "assistant"
            else None
        )

        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                display_response(message["content"])
            else:
                st.markdown(str(message["content"]))


# --------------------------------------------------
# QUESTION FORM
# --------------------------------------------------

st.markdown(
    '<div class="input-heading">Magtanong sa LalimSalita</div>',
    unsafe_allow_html=True,
)

with st.form(
    "question_form",
    clear_on_submit=True,
    enter_to_submit=True,
):
    question_col, send_col = st.columns(
        [5, 1],
        vertical_alignment="bottom",
    )

    with question_col:
        typed_question = st.text_input(
            "Tanong",
            placeholder="I-type ang iyong tanong dito...",
            label_visibility="collapsed",
        )

    with send_col:
        submitted = st.form_submit_button(
            "Ipadala",
            use_container_width=True,
        )


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

question = None

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

elif submitted:
    question = typed_question

if question and question.strip():
    question = question.strip()

    if st.session_state.active_chat is None:
        st.session_state.active_chat = create_conversation(
            st.session_state.visitor_id,
            question,
        )

    conversation_id = st.session_state.active_chat

    save_message(
        conversation_id,
        st.session_state.visitor_id,
        "user",
        question,
    )

    with st.spinner("Hinahanap ang kahulugan..."):
        try:
            result = answer_from_backend(question)

        except Exception as exc:
            print("LalimSalita backend error:", repr(exc))

            result = {
                "answer": (
                    "May problema sa pagkuha ng sagot. "
                    "Pakisubukang muli."
                ),
                "sources": [],
                "has_evidence": False,
            }

    save_message(
        conversation_id,
        st.session_state.visitor_id,
        "assistant",
        result,
    )

    st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        LalimSalita · Filipino Vocabulary Learning Companion
    </div>
    """,
    unsafe_allow_html=True,
)
