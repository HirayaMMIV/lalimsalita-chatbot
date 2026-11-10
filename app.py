
import base64
import html
import json
import sqlite3
import uuid

from datetime import datetime
from pathlib import Path

import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LalimSalita",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

LOGO_PATH = BASE_DIR / "lalimsalita_logo.png"
DB_PATH = BASE_DIR / "lalimsalita_chats.db"

FALLBACK = (
    "Walang sapat na impormasyon sa ibinigay na dokumento."
)

SUGGESTIONS = [
    "Ano ang kahulugan ng hilahil?",
    "Ano ang ibig sabihin ng dalamhati?",
    "Ano ang kahulugan ng nagpipighati?"
]


# =====================================================
# 2. DATABASE
# =====================================================

def get_connection():

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10
    )

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                visitor_id TEXT NOT NULL,
                title TEXT NOT NULL,
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
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
            )
        """)


def current_time():

    return datetime.now().isoformat(
        timespec="seconds"
    )


def create_conversation(visitor_id, question):

    conversation_id = str(uuid.uuid4())
    timestamp = current_time()

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO conversations
            (id, visitor_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            conversation_id,
            visitor_id,
            question[:45],
            timestamp,
            timestamp
        ))

    return conversation_id


def get_conversations(visitor_id):

    with get_connection() as conn:

        return conn.execute("""
            SELECT id, title, updated_at
            FROM conversations
            WHERE visitor_id = ?
            ORDER BY updated_at DESC
        """, (visitor_id,)).fetchall()


def owns_conversation(conversation_id, visitor_id):

    with get_connection() as conn:

        row = conn.execute("""
            SELECT 1
            FROM conversations
            WHERE id = ? AND visitor_id = ?
        """, (
            conversation_id,
            visitor_id
        )).fetchone()

    return row is not None


def save_message(
    conversation_id,
    role,
    content,
    visitor_id
):

    if not owns_conversation(
        conversation_id,
        visitor_id
    ):
        raise PermissionError(
            "Conversation is not accessible."
        )

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO messages
            (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            conversation_id,
            role,
            json.dumps(
                content,
                ensure_ascii=False
            ),
            current_time()
        ))

        conn.execute("""
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
        """, (
            current_time(),
            conversation_id
        ))


def get_messages(conversation_id, visitor_id):

    if not conversation_id:
        return []

    if not owns_conversation(
        conversation_id,
        visitor_id
    ):
        return []

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,)).fetchall()

    messages = []

    for row in rows:

        try:
            content = json.loads(
                row["content"]
            )

        except json.JSONDecodeError:
            content = row["content"]

        messages.append({
            "role": row["role"],
            "content": content
        })

    return messages


def delete_conversation(
    conversation_id,
    visitor_id
):

    if not owns_conversation(
        conversation_id,
        visitor_id
    ):
        return

    with get_connection() as conn:

        conn.execute("""
            DELETE FROM messages
            WHERE conversation_id = ?
        """, (conversation_id,))

        conn.execute("""
            DELETE FROM conversations
            WHERE id = ? AND visitor_id = ?
        """, (
            conversation_id,
            visitor_id
        ))


init_database()


# =====================================================
# 3. SESSION STATE
# =====================================================

# Development-only visitor identification.
# Replace with authenticated users before
# deploying to students.

visitor_param = st.query_params.get("visitor")

if "visitor_id" not in st.session_state:

    try:

        st.session_state.visitor_id = str(
            uuid.UUID(str(visitor_param))
        )

    except (
        ValueError,
        TypeError,
        AttributeError
    ):

        st.session_state.visitor_id = (
            str(uuid.uuid4())
        )


if visitor_param != st.session_state.visitor_id:

    st.query_params["visitor"] = (
        st.session_state.visitor_id
    )


if "active_chat" not in st.session_state:
    st.session_state.active_chat = None


if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


if "preview_mode" not in st.session_state:
    st.session_state.preview_mode = True


# =====================================================
# 4. LOGO HELPERS
# =====================================================

def get_logo_base64():

    if not LOGO_PATH.is_file():
        return None

    return base64.b64encode(
        LOGO_PATH.read_bytes()
    ).decode("utf-8")


def get_logo_html(
    css_class="brand-logo"
):

    encoded = get_logo_base64()

    if encoded:

        return (
            f'<img class="{css_class}" '
            f'src="data:image/png;base64,{encoded}" '
            f'alt="LalimSalita logo">'
        )

    return (
        f'<div class="{css_class} '
        f'logo-fallback">L</div>'
    )


def get_assistant_avatar():

    if LOGO_PATH.is_file():
        return str(LOGO_PATH)

    return None


def render_html(markup):

    cleaned = " ".join(
        line.strip()
        for line in markup.splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True
    )


# =====================================================
# 5. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: light;
}

html,
body,
.stApp {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #fcfbff;
    color: #293452;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 830px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================
   SIDEBAR
========================================== */

[data-testid="stSidebar"] {
    background: #f6f3ff;
    border-right: 1px solid #e6def9;
}

[data-testid="stSidebar"] * {
    color: #34345a;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 15px 0 6px;
}

.brand-logo {
    width: 50px;
    height: 50px;
    object-fit: contain;
    flex-shrink: 0;
}

.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #49369b;
}

.brand-caption {
    font-size: 12px;
    color: #827e9b;
    margin-bottom: 26px;
}

.sidebar-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    color: #827b9d;
    margin: 22px 0 10px;
}


/* ==========================================
   WELCOME SCREEN
========================================== */

.welcome {
    text-align: center;
    margin-top: 11vh;
    margin-bottom: 30px;
}

.welcome-logo {
    display: block;
    width: 94px;
    height: 94px;
    object-fit: contain;
    margin: 0 auto 20px;
}

.logo-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #7358e8;
    color: white !important;
    border-radius: 18px;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 32px;
}

.welcome-title {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(28px, 4vw, 37px);
    font-weight: 800;
    color: #34315e;
    margin-bottom: 10px;
}

.welcome-description {
    color: #85839e;
    font-size: 15px;
    line-height: 1.7;
}


/* ==========================================
   CHAT HEADER
========================================== */

.chat-header {
    padding: 6px 0 16px;
    border-bottom: 1px solid #eee9fa;
    margin-bottom: 20px;
}

.chat-header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 21px;
    font-weight: 800;
    color: #42358f;
}

.chat-header-subtitle {
    font-size: 12px;
    color: #9390a9;
    margin-top: 4px;
}


/* ==========================================
   BUTTONS
========================================== */

.stButton > button {
    background: white !important;
    border: 1px solid #e1d9fa !important;
    color: #5945b4 !important;
    border-radius: 13px !important;
    font-weight: 700 !important;
    min-height: 44px;
    white-space: normal;
}

.stButton > button:hover {
    background: #f1ecff !important;
    border-color: #ad99f0 !important;
}

[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    text-align: left;
}


/* ==========================================
   CHAT MESSAGES
========================================== */

[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 10px 0;
    margin-bottom: 10px;
}

[data-testid="stChatMessage"] p {
    color: #34405d;
    line-height: 1.75;
}


/* ==========================================
   QUESTION INPUT
========================================== */

.input-heading {
    font-family: 'Outfit', sans-serif;
    color: #50429b;
    font-weight: 700;
    font-size: 14px;
    margin: 20px 0 10px;
}

[data-testid="stForm"] {
    background: #f1edff !important;
    border: 1px solid #d9cef8 !important;
    border-radius: 19px !important;
    padding: 13px 15px !important;
    box-shadow: 0 9px 25px #7561be12;
}

[data-testid="stTextInput"] > div > div {
    background: #ffffff !important;
    border: 1px solid #d6cafa !important;
    border-radius: 12px !important;
}

[data-testid="stTextInput"] input {
    background: #ffffff !important;
    color: #28334f !important;
    -webkit-text-fill-color: #28334f !important;
    border-radius: 12px !important;
    font-size: 14px !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #9290a9 !important;
    opacity: 1;
}


/* ==========================================
   HIDE "PRESS ENTER TO SUBMIT FORM"
========================================== */

/*
   Hide Streamlit's input instruction.
   This does NOT disable Enter submission.
*/

[data-testid="InputInstructions"],
[data-testid="stInputInstructions"],
[data-testid="stTextInput"] [data-testid="InputInstructions"] {
    display: none !important;
}


/* ==========================================
   SUBMIT BUTTON
========================================== */

[data-testid="stFormSubmitButton"] button {
    background: #7257e9 !important;
    border: none !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    min-height: 43px !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #6045d5 !important;
    color: white !important;
}


/* ==========================================
   VOCABULARY CARDS
========================================== */

.word-card {
    background: white;
    border: 1px solid #e9e1fa;
    border-left: 5px solid #8a6bea;
    border-radius: 18px;
    padding: 20px 22px;
    margin: 10px 0 13px;
}

.word-title {
    font-family: 'Outfit', sans-serif;
    color: #5b43b9;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 9px;
}

.word-pos {
    display: inline-block;
    padding: 5px 12px;
    background: #e6f3ff;
    color: #346b9d;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.word-label {
    color: #6250a3;
    font-weight: 800;
    font-size: 13px;
    margin-top: 16px;
}

.word-text {
    color: #39435d;
    font-size: 14px;
    line-height: 1.75;
    margin-top: 5px;
}

.word-source {
    color: #8a7cbb;
    font-size: 12px;
    font-weight: 700;
    margin-top: 16px;
}


/* ==========================================
   RETRIEVED SOURCES
========================================== */

[data-testid="stExpander"] {
    background: #faf9ff;
    border: 1px solid #e8e3f5;
    border-radius: 13px;
}


/* ==========================================
   FOOTER
========================================== */

.footer {
    text-align: center;
    font-size: 12px;
    color: #a3a0b8;
    padding: 22px 0 10px;
}


/* ==========================================
   MOBILE RESPONSIVENESS
========================================== */

@media (max-width: 700px) {

    .welcome {
        margin-top: 5vh;
    }

    .block-container {
        padding-top: 1rem;
    }

    .word-card {
        padding: 16px;
    }

}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 6. RAG BACKEND
# =====================================================

def answer_from_backend(question):

    """
    UI preview mode.

    Connect your actual rag_backend.py
    in the non-preview branch.
    """

    if st.session_state.preview_mode:

        if "hilahil" in question.casefold():

            return {
                "answer": "",
                "word": "hilahil",
                "part_of_speech": "pangngalan",
                "definition": (
                    "Dusa, bagabag, o malalim na "
                    "pagdurusa at ligalig."
                ),
                "context": (
                    "Mga paghihirap at alalahaning "
                    "dinanas dahil sa pag-ibig."
                ),
                "sources": [
                    {
                        "page": 5,
                        "text": (
                            "WORD: hilahil\n"
                            "PART_OF_SPEECH: pangngalan\n"
                            "DEFINITION: dusa, bagabag, "
                            "o malalim na pagdurusa at ligalig\n"
                            "CONTEXTUAL_MEANING: mga paghihirap "
                            "at alalahaning dinanas dahil sa pag-ibig"
                        )
                    }
                ],
                "has_evidence": True,
                "preview": True
            }

        return {
            "answer": FALLBACK,
            "sources": [],
            "has_evidence": False,
            "preview": True
        }

    # =================================================
    # CONNECT YOUR ACTUAL RAG BACKEND HERE
    # =================================================
    #
    # Example only:
    #
    # from rag_backend import ask_question
    # return ask_question(question)
    #

    return {
        "answer": "Hindi pa nakakonekta ang RAG backend.",
        "sources": [],
        "has_evidence": False
    }


# =====================================================
# 7. DISPLAY CHATBOT RESPONSE
# =====================================================

def display_response(result):

    if not isinstance(result, dict):

        st.markdown(str(result))
        return

    answer = str(
        result.get("answer") or ""
    ).strip()

    sources = result.get("sources") or []

    unsupported = (
        answer.casefold() == FALLBACK.casefold()
    )

    has_evidence = (
        bool(result.get("has_evidence", False))
        and not unsupported
    )

    if (
        not unsupported
        and result.get("word")
        and result.get("definition")
    ):

        word = html.escape(
            str(result["word"])
        )

        part = html.escape(
            str(result.get("part_of_speech") or "")
        )

        definition = html.escape(
            str(result["definition"])
        )

        context = html.escape(
            str(result.get("context") or "")
        )

        page = "—"

        if (
            sources
            and isinstance(sources[0], dict)
        ):

            page = html.escape(
                str(
                    sources[0].get("page", "—")
                )
            )

        render_html(f"""
        <div class="word-card">

            <div class="word-title">
                {word}
            </div>

            <span class="word-pos">
                {part}
            </span>

            <div class="word-label">
                Kahulugan
            </div>

            <div class="word-text">
                {definition}
            </div>

            <div class="word-label">
                Sa konteksto
            </div>

            <div class="word-text">
                {context}
            </div>

            <div class="word-source">
                Talasalitaan · Pahina {page}
            </div>

        </div>
        """)

    if answer:
        st.markdown(answer)

    # Do not show sources for unsupported questions.
    if has_evidence and sources:

        with st.expander(
            f"Tingnan ang mga sanggunian ({len(sources)})"
        ):

            for index, source in enumerate(
                sources,
                start=1
            ):

                if isinstance(source, dict):

                    page = source.get(
                        "page",
                        "—"
                    )

                    source_text = source.get(
                        "text",
                        ""
                    )

                    distance = source.get(
                        "distance"
                    )

                else:

                    page = "—"
                    source_text = str(source)
                    distance = None

                st.markdown(
                    f"**Sanggunian {index} · Pahina {page}**"
                )

                if isinstance(
                    distance,
                    (int, float)
                ):

                    st.caption(
                        f"Cosine distance: {distance:.4f}"
                    )

                st.code(
                    str(source_text),
                    language=None
                )

                if index < len(sources):
                    st.divider()

    if result.get("preview"):

        st.caption(
            "Preview mode lamang — hindi live retrieval."
        )


# =====================================================
# 8. SIDEBAR
# =====================================================

with st.sidebar:

    logo_html = get_logo_html(
        "brand-logo"
    )

    render_html(f"""
    <div class="brand">
        {logo_html}
        <div class="brand-name">
            LalimSalita
        </div>
    </div>

    <div class="brand-caption">
        Kasama mo sa pag-unawa ng Filipino
    </div>
    """)

    if st.button(
        "Bagong usapan",
        use_container_width=True
    ):

        st.session_state.active_chat = None
        st.session_state.pending_question = None

        st.rerun()

    render_html("""
    <div class="sidebar-label">
        MGA NAKARAANG USAPAN
    </div>
    """)

    conversations = get_conversations(
        st.session_state.visitor_id
    )

    if not conversations:

        st.caption(
            "Wala pang nakaraang usapan."
        )

    for conversation in conversations:

        chat_col, delete_col = st.columns(
            [5, 1],
            vertical_alignment="center"
        )

        with chat_col:

            if st.button(
                conversation["title"],
                key=f"open_{conversation['id']}",
                use_container_width=True
            ):

                st.session_state.active_chat = (
                    conversation["id"]
                )

                st.session_state.pending_question = None

                st.rerun()

        with delete_col:

            if st.button(
                "×",
                key=f"delete_{conversation['id']}",
                help="Burahin ang usapan"
            ):

                delete_conversation(
                    conversation["id"],
                    st.session_state.visitor_id
                )

                if (
                    st.session_state.active_chat
                    == conversation["id"]
                ):

                    st.session_state.active_chat = None

                st.rerun()

    render_html("""
    <div class="sidebar-label">
        MGA SETTING
    </div>
    """)

    with st.expander(
        "Retrieval settings"
    ):

        st.write(
            "Chunk size: **500** characters"
        )

        st.write(
            "Chunk overlap: **100** characters"
        )

        st.write(
            "Top-K: **3**"
        )

        st.session_state.preview_mode = st.toggle(
            "Preview mode",
            value=st.session_state.preview_mode
        )

    st.caption(
        "Talasalitaan Aralin 1–29"
    )


# =====================================================
# 9. LOAD CURRENT CONVERSATION
# =====================================================

messages = get_messages(
    st.session_state.active_chat,
    st.session_state.visitor_id
)


# =====================================================
# 10. WELCOME SCREEN
# =====================================================

if not messages:

    welcome_logo = get_logo_html(
        "welcome-logo"
    )

    render_html(f"""
    <div class="welcome">

        {welcome_logo}

        <div class="welcome-title">
            Kumusta! Ako si LalimSalita.
        </div>

        <div class="welcome-description">
            Anong salitang Filipino ang gusto mong
            mas maintindihan ngayon?
        </div>

    </div>
    """)

    suggestion_columns = st.columns(3)

    for column, suggestion in zip(
        suggestion_columns,
        SUGGESTIONS
    ):

        with column:

            if st.button(
                suggestion,
                use_container_width=True
            ):

                st.session_state.pending_question = (
                    suggestion
                )

                st.rerun()


# =====================================================
# 11. DISPLAY CONVERSATION
# =====================================================

else:

    render_html("""
    <div class="chat-header">

        <div class="chat-header-title">
            LalimSalita
        </div>

        <div class="chat-header-subtitle">
            Filipino Vocabulary Learning Companion
        </div>

    </div>
    """)

    for message in messages:

        role = message["role"]

        # Use an actual image path for assistant avatar.
        # Use Streamlit's default user avatar.

        avatar = (
            get_assistant_avatar()
            if role == "assistant"
            else None
        )

        with st.chat_message(
            role,
            avatar=avatar
        ):

            if role == "assistant":

                display_response(
                    message["content"]
                )

            else:

                st.markdown(
                    str(message["content"])
                )


# =====================================================
# 12. QUESTION INPUT
# =====================================================

render_html("""
<div class="input-heading">
    Magtanong sa LalimSalita
</div>
""")

with st.form(
    "question_form",
    clear_on_submit=True,
    enter_to_submit=True
):

    question_col, send_col = st.columns(
        [5, 1],
        vertical_alignment="bottom"
    )

    with question_col:

        typed_question = st.text_input(
            "Tanong",
            placeholder="I-type ang iyong tanong dito...",
            label_visibility="collapsed"
        )

    with send_col:

        submitted = st.form_submit_button(
            "Ipadala",
            use_container_width=True
        )


# =====================================================
# 13. PROCESS QUESTION
# =====================================================

question = None

if st.session_state.pending_question:

    question = st.session_state.pending_question

    st.session_state.pending_question = None

elif submitted:

    question = typed_question


if question and question.strip():

    question = question.strip()

    if st.session_state.active_chat is None:

        st.session_state.active_chat = (
            create_conversation(
                st.session_state.visitor_id,
                question
            )
        )

    conversation_id = (
        st.session_state.active_chat
    )

    # Save the user's question.
    save_message(
        conversation_id,
        "user",
        question,
        st.session_state.visitor_id
    )

    # Generate the assistant response.
    with st.spinner(
        "Hinahanap ang kahulugan..."
    ):

        try:

            result = answer_from_backend(
                question
            )

        except Exception as exc:

            print(
                "RAG backend error:",
                exc
            )

            result = {
                "answer": (
                    "May problema sa pagkuha ng sagot. "
                    "Pakisubukang muli."
                ),
                "sources": [],
                "has_evidence": False
            }

    # Save the assistant's response.
    save_message(
        conversation_id,
        "assistant",
        result,
        st.session_state.visitor_id
    )

    # Refresh to display the conversation.
    st.rerun()


# =====================================================
# 14. FOOTER
# =====================================================

render_html("""
<div class="footer">
    LalimSalita · Filipino Vocabulary Learning Companion
</div>
""")
