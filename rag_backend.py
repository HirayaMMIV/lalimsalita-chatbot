
import os
import json
import logging
import re
import gc
import threading
from pathlib import Path

# Limit numerical library threads before importing NumPy.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import requests
import onnxruntime as ort

from tokenizers import Tokenizer
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "retrieval_data"
MODEL_DIR = BASE_DIR / "onnx_model"

CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

MODEL_PATH = MODEL_DIR / "model_int8.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "hirayammiv/lalimsalita-e5-onnx",
)

TOP_K = 3
MAX_QUERY_TOKENS = 128

FALLBACK = (
    "Walang sapat na impormasyon "
    "sa ibinigay na dokumento."
)

FALLBACK_ANSWER = FALLBACK

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

logger = logging.getLogger("lalimsalita.backend")
logging.basicConfig(level=logging.INFO)

# Prevent simultaneous requests from loading
# multiple copies of the ONNX model.
_MODEL_LOCK = threading.Lock()
_MODEL_SESSION = None
_TOKENIZER = None


# ============================================================
# LOAD PRECOMPUTED RETRIEVAL DATA
# ============================================================

def load_retrieval_data():

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing retrieval file: {CHUNKS_PATH}"
        )

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing retrieval file: {EMBEDDINGS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    if isinstance(chunks, dict):
        chunks = chunks.get(
            "chunks",
            chunks.get("documents", []),
        )

    if not isinstance(chunks, list):
        raise ValueError(
            "chunks.json must contain a list."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings.npy must be a 2D array."
        )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk and embedding counts do not match. "
            f"Chunks: {len(chunks)}, "
            f"Embeddings: {len(embeddings)}"
        )

    # Normalize existing embeddings for cosine similarity.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    embeddings /= np.maximum(
        norms,
        1e-12,
    )

    logger.info(
        "Loaded %d precomputed chunks.",
        len(chunks),
    )

    return chunks, embeddings


CHUNKS, EMBEDDINGS = load_retrieval_data()


# ============================================================
# DOWNLOAD MODEL FILES IF MISSING
# ============================================================

def ensure_model_files():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        "model_int8.onnx",
        "tokenizer.json",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (MODEL_DIR / filename).exists()
    ]

    if not missing_files:
        logger.info(
            "Using existing local ONNX model files."
        )
        return

    # Import only when downloading is required.
    from huggingface_hub import hf_hub_download

    token = os.getenv(
        "HF_TOKEN",
        "",
    ).strip() or None

    for filename in missing_files:

        logger.info(
            "Downloading %s from Hugging Face...",
            filename,
        )

        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=str(MODEL_DIR),
            token=token,
        )

    logger.info(
        "Required ONNX model files are available."
    )


# ============================================================
# MEMORY-OPTIMIZED ONNX SESSION
# ============================================================

def get_model():

    global _MODEL_SESSION
    global _TOKENIZER

    if (
        _MODEL_SESSION is not None
        and _TOKENIZER is not None
    ):
        return _MODEL_SESSION, _TOKENIZER

    with _MODEL_LOCK:

        if (
            _MODEL_SESSION is not None
            and _TOKENIZER is not None
        ):
            return _MODEL_SESSION, _TOKENIZER

        ensure_model_files()

        logger.info(
            "Loading lightweight tokenizer..."
        )

        tokenizer = Tokenizer.from_file(
            str(TOKENIZER_PATH)
        )

        # E5 requires the query prefix.
        # Tokenizer truncation prevents unnecessarily
        # long inputs from increasing memory use.
        tokenizer.enable_truncation(
            max_length=MAX_QUERY_TOKENS
        )

        logger.info(
            "Initializing memory-optimized ONNX Runtime..."
        )

        options = ort.SessionOptions()

        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

        # Reduce additional CPU memory allocations.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False

        session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        logger.info(
            "ONNX model initialized successfully."
        )

        _TOKENIZER = tokenizer
        _MODEL_SESSION = session

        return _MODEL_SESSION, _TOKENIZER


# ============================================================
# GENERATE QUERY EMBEDDING
# ============================================================

def get_query_embedding(question):

    session, tokenizer = get_model()

    question = str(question).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    prefixed_question = "query: " + question

    encoded = tokenizer.encode(
        prefixed_question,
        add_special_tokens=True,
    )

    input_ids = np.asarray(
        [encoded.ids],
        dtype=np.int64,
    )

    attention_mask = np.asarray(
        [encoded.attention_mask],
        dtype=np.int64,
    )

    token_type_ids = np.zeros_like(
        input_ids,
        dtype=np.int64,
    )

    available_inputs = {
        item.name
        for item in session.get_inputs()
    }

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if "token_type_ids" in available_inputs:
        inputs["token_type_ids"] = token_type_ids

    # Execute inference once.
    outputs = session.run(
        None,
        inputs,
    )

    # E5 ONNX exports commonly return token embeddings:
    # [batch, sequence_length, hidden_dimension].
    token_embeddings = np.asarray(
        outputs[0],
        dtype=np.float32,
    )

    if token_embeddings.ndim == 3:

        # Attention-mask-aware mean pooling.
        mask = attention_mask[
            :, :token_embeddings.shape[1]
        ].astype(np.float32)

        expanded_mask = np.expand_dims(
            mask,
            axis=-1,
        )

        summed = np.sum(
            token_embeddings * expanded_mask,
            axis=1,
        )

        counts = np.maximum(
            np.sum(expanded_mask, axis=1),
            1e-12,
        )

        vector = summed / counts
        vector = vector[0]

    elif token_embeddings.ndim == 2:

        if token_embeddings.shape[0] != 1:
            raise ValueError(
                "Unexpected ONNX embedding output shape."
            )

        vector = token_embeddings[0]

    else:
        raise ValueError(
            "Unsupported ONNX output shape: "
            f"{token_embeddings.shape}"
        )

    if vector.shape[0] != EMBEDDINGS.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch. "
            f"Query: {vector.shape[0]}, "
            f"Stored: {EMBEDDINGS.shape[1]}"
        )

    norm = np.linalg.norm(vector)

    if norm <= 0:
        raise ValueError(
            "Query embedding has zero magnitude."
        )

    vector = vector / norm

    # Release temporary arrays.
    del outputs
    del token_embeddings

    return vector.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# CHUNK UTILITIES
# ============================================================

def get_chunk_text(chunk):

    if isinstance(chunk, str):
        return chunk

    if not isinstance(chunk, dict):
        return str(chunk)

    for key in (
        "text",
        "page_content",
        "content",
        "chunk_text",
    ):
        value = chunk.get(key)

        if value:
            return str(value)

    return ""


def get_chunk_page(chunk):

    if not isinstance(chunk, dict):
        return "Unknown"

    metadata = chunk.get("metadata") or {}

    if not isinstance(metadata, dict):
        metadata = {}

    for key in (
        "page",
        "page_number",
        "source_page",
    ):

        if chunk.get(key) is not None:
            return chunk[key]

        if metadata.get(key) is not None:
            return metadata[key]

    return "Unknown"


# ============================================================
# EXACT VOCABULARY MATCHING
# ============================================================

def normalize_text(value):

    value = str(value).casefold()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    return " ".join(
        value.split()
    )


def get_vocabulary_words(chunk):

    if not isinstance(chunk, dict):
        return []

    words = []

    for key in (
        "word",
        "WORD",
        "vocabulary_word",
        "term",
    ):

        value = chunk.get(key)

        if isinstance(value, str) and value.strip():
            words.append(value.strip())

    metadata = chunk.get("metadata") or {}

    if isinstance(metadata, dict):

        for key in (
            "word",
            "WORD",
            "vocabulary_word",
            "term",
        ):

            value = metadata.get(key)

            if isinstance(value, str) and value.strip():
                words.append(value.strip())

    text = get_chunk_text(chunk)

    matches = re.findall(
        r"(?im)^\s*WORD\s*:\s*([^\n\r]+)",
        text,
    )

    words.extend(matches)

    return list(
        dict.fromkeys(words)
    )


def exact_match_score(question, chunk):

    normalized_question = normalize_text(
        question
    )

    words = get_vocabulary_words(chunk)

    for word in words:

        normalized_word = normalize_text(word)

        if not normalized_word:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_word)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized_question,
        ):
            return 1.0

    return 0.0


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve(question, top_k=TOP_K):

    question = str(question).strip()

    if not question:
        return []

    query_vector = get_query_embedding(
        question
    )

    similarities = EMBEDDINGS @ query_vector

    results = []

    for index, chunk in enumerate(CHUNKS):

        text = get_chunk_text(chunk)

        if not text.strip():
            continue

        similarity = float(
            similarities[index]
        )

        exact = exact_match_score(
            question,
            chunk,
        )

        # Prioritize vocabulary matches while
        # retaining semantic similarity ranking.
        score = similarity + (0.25 * exact)

        results.append({
            "chunk_id": index,
            "page": get_chunk_page(chunk),
            "text": text,
            "similarity": similarity,
            "distance": 1.0 - similarity,
            "match_type": (
                "exact vocabulary"
                if exact
                else "semantic"
            ),
            "_score": score,
        })

    results.sort(
        key=lambda item: item["_score"],
        reverse=True,
    )

    selected = results[:top_k]

    for rank, result in enumerate(
        selected,
        start=1,
    ):

        result["rank"] = rank
        result.pop("_score", None)

    logger.info(
        "Retrieved %d chunks for query.",
        len(selected),
    )

    return selected


# ============================================================
# OUT-OF-SCOPE FILTER
# ============================================================

def is_out_of_scope(question):

    raw = str(question).casefold()
    normalized = normalize_text(question)

    math_patterns = [
        r"\b\d+\s*[\+\-\*/×÷]\s*\d+\b",
        r"\b\d+\s*(plus|minus|times)\s*\d+\b",
    ]

    for pattern in math_patterns:
        if re.search(pattern, raw):
            return True

    # Reject clearly unrelated factual questions.
    out_of_scope_patterns = [
        r"kasalukuyang pangulo",
        r"president of the philippines",
        r"current president",
    ]

    for pattern in out_of_scope_patterns:
        if re.search(pattern, normalized):
            return True

    return False


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved,
    api_key,
):

    if not retrieved:
        return FALLBACK

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f"[Source {item['rank']} | "
            f"Page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
Ikaw si LalimSalita, isang Filipino vocabulary
learning assistant.

Gamitin lamang ang impormasyong nasa
retrieved source context.

Mga tuntunin:

1. Sagutin sa malinaw at natural na Filipino.

2. Ipaliwanag ang kahulugan ng salita.

3. Isama ang konteksto kung sinusuportahan
   ito ng dokumento.

4. Huwag mag-imbento ng kahulugan,
   halimbawa, o sanggunian.

5. Kung hindi sinusuportahan ng context
   ang sagot, sabihin nang eksakto:

Walang sapat na impormasyon sa ibinigay na dokumento.

6. Huwag sagutin ang mga tanong na
   wala sa saklaw ng talasalitaan.
"""

    user_prompt = f"""
TANONG:
{question}

RETRIEVED CONTEXT:
{context}

Sagutin ang tanong gamit lamang ang context.
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": 450,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    answer = data[
        "choices"
    ][0]["message"]["content"]

    return str(
        answer or FALLBACK
    ).strip()


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def rag_query(question, api_key=None):

    question = str(question).strip()

    if not question:
        return {
            "answer": FALLBACK,
            "retrieved": [],
        }

    if is_out_of_scope(question):
        return {
            "answer": FALLBACK,
            "retrieved": [],
        }

    if not api_key:
        api_key = os.getenv(
            "GROQ_API_KEY",
            "",
        ).strip()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing."
        )

    retrieved = retrieve(
        question,
        top_k=TOP_K,
    )

    answer = generate_answer(
        question,
        retrieved,
        api_key,
    )

    if FALLBACK.casefold() in answer.casefold():
        retrieved = []

    return {
        "answer": answer,
        "retrieved": retrieved,
    }
