
"""LalimSalita RAG backend with hybrid vocabulary retrieval."""

import os
import re
from pathlib import Path
from functools import lru_cache

# ============================================================
# 1. CONFIGURATION AND CACHE
# ============================================================

BASE = Path(__file__).resolve().parent

PDF_PATH = BASE / "data" / "Talasalitaan Dataset (1).pdf"

CACHE_DIR = BASE / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configure cache locations before importing ML libraries.
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(
    CACHE_DIR / "sentence_transformers"
)
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")

import chromadb
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "intfloat/multilingual-e5-small"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = 3

FALLBACK_ANSWER = (
    "Walang sapat na impormasyon sa ibinigay na dokumento."
)

SYSTEM_PROMPT = """
You are LalimSalita, a Filipino vocabulary assistant.

Answer in Filipino using ONLY the provided PDF excerpts.

Rules:
1. Explain the meaning of the vocabulary term clearly.
2. Use the DEFINITION and CONTEXTUAL_MEANING fields when available.
3. Cite the PDF page number, for example (Pahina 5).
4. Do not invent vocabulary definitions or references.
5. If the retrieved excerpts do not support the answer, say exactly:
Walang sapat na impormasyon sa ibinigay na dokumento.
"""


# ============================================================
# 2. PDF TEXT FORMATTING
# ============================================================

def format_vocabulary_text(raw_text):
    """Normalize extracted vocabulary text."""

    text = re.sub(r"\s+", " ", raw_text or "").strip()

    text = re.sub(
        r"\bARALIN\s+(\d+)\b",
        r"\n\nARALIN \1\n",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bVOCABULARY\s+ENTRY\s+(\d+)\b",
        r"\n\nVOCABULARY ENTRY \1\n",
        text,
        flags=re.IGNORECASE
    )

    fields = (
        "WORD",
        "PART_OF_SPEECH",
        "DEFINITION",
        "CONTEXTUAL_MEANING",
        "STANZA",
        "SOURCE"
    )

    for field in fields:
        text = re.sub(
            rf"\s*\b({field})\s*:\s*",
            r"\n\1: ",
            text,
            flags=re.IGNORECASE
        )

    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ============================================================
# 3. PDF PROCESSING
# ============================================================

def process_pdf():

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF dataset not found: {PDF_PATH}"
        )

    reader = PdfReader(str(PDF_PATH))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    vocabulary_index = {}

    for page_no, page in enumerate(reader.pages, start=1):

        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            print(
                f"Warning: Could not extract page "
                f"{page_no}: {exc}"
            )
            continue

        cleaned_text = format_vocabulary_text(raw_text)

        if not cleaned_text:
            continue

        page_chunks = splitter.split_text(cleaned_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "id": f"p{page_no}_c{chunk_index}",
                "page": page_no,
                "text": chunk_text
            })

        # Extract complete vocabulary entries for
        # exact headword matching.
        entries = re.split(
            r"(?im)(?=^VOCABULARY ENTRY[ \t]+\d+[ \t]*$)",
            cleaned_text
        )

        for entry in entries:

            match = re.search(
                r"(?im)^WORD:[ \t]*([^\n]+)",
                entry
            )

            if not match:
                continue

            word = match.group(1).strip().lower()

            if word:
                vocabulary_index.setdefault(
                    word, []
                ).append({
                    "page": page_no,
                    "text": entry.strip()
                })

    if not chunks:
        raise ValueError(
            "No extractable vocabulary text found in the PDF. "
            "Check that the PDF is valid and contains "
            "selectable text."
        )

    print("\n========== DOCUMENT PROCESSING ==========")
    print(f"PDF: {PDF_PATH.name}")
    print(f"PDF pages: {len(reader.pages)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Total semantic chunks: {len(chunks)}")
    print(f"Indexed vocabulary words: {len(vocabulary_index)}")
    print("=========================================\n")

    return chunks, vocabulary_index


# ============================================================
# 4. EMBEDDINGS AND CHROMADB
# ============================================================

@lru_cache(maxsize=1)
def load_pipeline():

    print("Initializing LalimSalita RAG pipeline...")

    encoder = SentenceTransformer(
        EMBED_MODEL,
        cache_folder=str(CACHE_DIR / "sentence_transformers")
    )

    chunks, vocabulary_index = process_pdf()

    client = chromadb.EphemeralClient()

    collection = client.create_collection(
        name="lalimsalita_live",
        metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(chunks), 100):

        batch = chunks[start:start + 100]

        passages = [
            "passage: " + item["text"]
            for item in batch
        ]

        vectors = encoder.encode(
            passages,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )

        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[
                {"page": item["page"]}
                for item in batch
            ],
            embeddings=vectors.tolist()
        )

    print(f"Indexed ChromaDB chunks: {collection.count()}")
    print("LalimSalita pipeline initialized successfully.")

    return encoder, collection, vocabulary_index


# ============================================================
# 5. EXACT VOCABULARY MATCHING
# ============================================================

def find_exact_vocabulary(question, vocabulary_index):

    question_normalized = re.sub(
        r"\s+",
        " ",
        question.lower()
    ).strip()

    matches = []

    for word in sorted(
        vocabulary_index.keys(),
        key=len,
        reverse=True
    ):

        pattern = (
            r"(?<!\w)"
            + re.escape(word)
            + r"(?!\w)"
        )

        if re.search(pattern, question_normalized):

            for entry in vocabulary_index[word]:
                matches.append({
                    "page": entry["page"],
                    "text": entry["text"],
                    "match_type": "exact"
                })

    return matches


# ============================================================
# 6. HYBRID RETRIEVAL
# ============================================================

def retrieve(question):

    encoder, collection, vocabulary_index = load_pipeline()

    query_vector = encoder.encode(
        ["query: " + question],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0].tolist()

    # A. Semantic retrieval
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(TOP_K, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    semantic_docs = []

    for doc, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        semantic_docs.append({
            "page": metadata["page"],
            "text": doc,
            "distance": float(distance),
            "match_type": "semantic"
        })

    # B. Exact vocabulary lookup
    exact_docs = find_exact_vocabulary(
        question,
        vocabulary_index
    )

    if exact_docs:

        exact_vectors = encoder.encode(
            [
                "passage: " + doc["text"]
                for doc in exact_docs
            ],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        query_array = np.asarray(
            query_vector,
            dtype=np.float32
        )

        for doc, vector in zip(exact_docs, exact_vectors):

            similarity = float(
                np.dot(query_array, vector)
            )

            doc["distance"] = max(
                0.0,
                1.0 - similarity
            )

    # C. Exact matches first, followed by semantic results
    combined = []
    seen = set()

    for doc in exact_docs + semantic_docs:

        key = (
            doc["page"],
            doc["text"].strip()
        )

        if key in seen:
            continue

        seen.add(key)
        combined.append(doc)

        if len(combined) >= TOP_K:
            break

    for i, doc in enumerate(combined):
        doc["rank"] = i + 1

    print("\n========== HYBRID RETRIEVAL ==========")
    print(f"Question: {question}")

    for doc in combined:
        print(
            f"Rank {doc['rank']} | "
            f"Page {doc['page']} | "
            f"Type {doc['match_type']} | "
            f"Distance {doc['distance']:.4f}"
        )

    print("======================================\n")

    return combined


# ============================================================
# 7. GROQ RESPONSE GENERATION
# ============================================================

def rag_query(question, api_key):

    docs = retrieve(question)

    if not docs:
        return {
            "answer": FALLBACK_ANSWER,
            "retrieved": []
        }

    context = "\n\n".join(
        (
            f"[Source: page {doc['page']}, "
            f"rank {doc['rank']}]\n"
            f"{doc['text']}"
        )
        for doc in docs
    )

    groq_client = Groq(api_key=api_key)

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_completion_tokens=500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context}\n\n"
                    f"QUESTION: {question}"
                )
            }
        ]
    )

    answer = completion.choices[0].message.content or ""

    if not answer.strip():
        answer = FALLBACK_ANSWER

    return {
        "answer": answer.strip(),
        "retrieved": docs
    }
