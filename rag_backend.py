
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from groq import Groq
from transformers import AutoTokenizer


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "onnx_model"
INDEX_DIR = BASE_DIR / "retrieval_data"

MODEL_PATH = MODEL_DIR / "model_int8.onnx"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

TOP_K = 3

FALLBACK_ANSWER = (
    "Walang sapat na impormasyon sa ibinigay na dokumento."
)

SYSTEM_PROMPT = f"""
You are LalimSalita, a Filipino vocabulary learning assistant
for Grade 8 students.

You answer questions about Filipino vocabulary using only
the provided Talasalitaan document excerpts.

RULES:

1. Respond in clear and student-friendly Filipino.

2. Use only information supported by the retrieved excerpts.

3. Use DEFINITION and CONTEXTUAL_MEANING when available.

4. Explain the meaning and context of the requested word.

5. Cite the supporting PDF page as (Pahina X).

6. Never invent definitions, contextual meanings, or citations.

7. Do not answer unrelated questions using outside knowledge.

8. If there is insufficient evidence, respond with exactly:

{FALLBACK_ANSWER}

9. Keep explanations concise and understandable.

10. Do not treat every retrieved excerpt as relevant.
    Verify that the excerpt actually supports the question.
"""

WORD_PATTERN = re.compile(
    r"(?im)^WORD:\s*([^\n]+)"
)


# ==========================================
# LOAD LIGHTWEIGHT RAG PIPELINE
# ==========================================

def make_session_options():
    options = ort.SessionOptions()

    # Limit CPU thread usage for small Render instances.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    # Avoid unnecessary memory allocations.
    options.enable_mem_pattern = False

    return options


@lru_cache(maxsize=1)
def load_pipeline():
    """
    Load the quantized ONNX model and saved
    vocabulary embeddings only once per process.
    """

    print(
        "Initializing lightweight LalimSalita RAG...",
        flush=True,
    )

    required_files = [
        MODEL_PATH,
        EMBEDDINGS_PATH,
        CHUNKS_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    print("Loading quantized ONNX model...", flush=True)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=make_session_options(),
        providers=["CPUExecutionProvider"],
    )

    print("Loading precomputed embeddings...", flush=True)

    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    ).astype(np.float32)

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match chunks."
        )

    # Normalize document vectors for cosine similarity.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    embeddings /= np.maximum(norms, 1e-12)

    # Index vocabulary headwords.
    vocabulary_index = {}

    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")

        for match in WORD_PATTERN.finditer(text):
            word = match.group(1).strip().casefold()

            if word:
                vocabulary_index.setdefault(
                    word,
                    [],
                ).append(index)

    print(
        f"LalimSalita ready: {len(chunks)} chunks.",
        flush=True,
    )

    return (
        tokenizer,
        session,
        embeddings,
        chunks,
        vocabulary_index,
    )


# ==========================================
# QUERY EMBEDDING USING ONNX
# ==========================================

def embed_question(question, tokenizer, session):
    """
    Generate a normalized multilingual E5
    query embedding using ONNX Runtime.
    """

    encoded = tokenizer(
        ["query: " + question],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    model_input_names = {
        item.name
        for item in session.get_inputs()
    }

    inputs = {
        key: np.asarray(
            value,
            dtype=np.int64,
        )
        for key, value in encoded.items()
        if key in model_input_names
    }

    missing_inputs = model_input_names - set(inputs)

    if missing_inputs:
        raise RuntimeError(
            f"Missing ONNX inputs: {missing_inputs}"
        )

    outputs = session.run(None, inputs)

    # Sentence Transformers E5 uses attention-mask
    # mean pooling, followed by L2 normalization.
    token_embeddings = outputs[0]

    attention_mask = encoded[
        "attention_mask"
    ][..., None].astype(np.float32)

    summed_embeddings = (
        token_embeddings * attention_mask
    ).sum(axis=1)

    token_counts = np.maximum(
        attention_mask.sum(axis=1),
        1e-12,
    )

    pooled = summed_embeddings / token_counts

    pooled /= np.maximum(
        np.linalg.norm(
            pooled,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    return pooled[0].astype(np.float32)


# ==========================================
# HYBRID RETRIEVAL
# ==========================================

def retrieve(question):
    """
    Retrieve up to three vocabulary excerpts.

    Priority:
    1. Exact vocabulary headword match
    2. Semantic cosine similarity
    """

    (
        tokenizer,
        session,
        embeddings,
        chunks,
        vocabulary_index,
    ) = load_pipeline()

    query_embedding = embed_question(
        question,
        tokenizer,
        session,
    )

    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Embedding dimensions do not match."
        )

    similarities = embeddings @ query_embedding

    normalized_question = question.casefold()

    exact_words = []

    for word in vocabulary_index:
        pattern = (
            r"(?<!\w)"
            + re.escape(word)
            + r"(?!\w)"
        )

        if re.search(pattern, normalized_question):
            exact_words.append(word)

    exact_words.sort(
        key=len,
        reverse=True,
    )

    selected = []
    seen_indices = set()

    # --------------------------------------
    # Exact headword matching
    # --------------------------------------

    for word in exact_words:
        candidates = vocabulary_index[word]

        candidates = sorted(
            candidates,
            key=lambda index: (
                "DEFINITION:" not in chunks[index]["text"],
                -float(similarities[index]),
            ),
        )

        for index in candidates:
            if index not in seen_indices:
                selected.append(
                    (index, "exact vocabulary")
                )
                seen_indices.add(index)
                break

        if len(selected) >= TOP_K:
            break

    # --------------------------------------
    # Semantic retrieval
    # --------------------------------------

    ranked_indices = np.argsort(-similarities)

    for index in ranked_indices:
        index = int(index)

        if len(selected) >= TOP_K:
            break

        if index not in seen_indices:
            selected.append(
                (index, "semantic")
            )
            seen_indices.add(index)

    # --------------------------------------
    # Format retrieved results
    # --------------------------------------

    results = []

    for rank, (index, method) in enumerate(
        selected,
        start=1,
    ):
        chunk = chunks[index]

        results.append({
            "rank": rank,
            "page": chunk["page"],
            "text": chunk["text"],
            "distance": float(
                max(
                    0.0,
                    1.0 - similarities[index],
                )
            ),
            "match_type": method,
            "retrieval_method": method,
        })

    print(
        f"Question: {question}",
        flush=True,
    )

    for result in results:
        print(
            f"Rank {result['rank']} | "
            f"Page {result['page']} | "
            f"{result['match_type']} | "
            f"Distance {result['distance']:.4f}",
            flush=True,
        )

    return results


# ==========================================
# GROQ RESPONSE GENERATION
# ==========================================

def rag_query(question, api_key):
    """
    Retrieve supporting vocabulary excerpts
    and generate a Filipino explanation.
    """

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    question = str(question or "").strip()

    if not question:
        return {
            "answer": FALLBACK_ANSWER,
            "retrieved": [],
        }

    retrieved_docs = retrieve(question)

    if not retrieved_docs:
        return {
            "answer": FALLBACK_ANSWER,
            "retrieved": [],
        }

    context = "\n\n".join(
        (
            f"[Source: PDF page {doc['page']}]\n"
            f"{doc['text']}"
        )
        for doc in retrieved_docs
    )

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_completion_tokens=500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context}\n\n"
                    f"STUDENT QUESTION: {question}"
                ),
            },
        ],
    )

    answer = (
        completion.choices[0].message.content or ""
    ).strip()

    if not answer:
        answer = FALLBACK_ANSWER

    # Hide source chunks when the chatbot
    # cannot answer the question.
    if answer.casefold() == FALLBACK_ANSWER.casefold():
        return {
            "answer": FALLBACK_ANSWER,
            "retrieved": [],
        }

    return {
        "answer": answer,
        "retrieved": retrieved_docs,
    }
