
"""
LalimSalita RAG Backend

PDF-based Filipino vocabulary question answering.

Components:
- pypdf for document extraction
- RecursiveCharacterTextSplitter
- multilingual-e5-small embeddings
- ChromaDB cosine retrieval
- Exact vocabulary matching
- Groq for grounded Filipino responses
"""

import os
import re
from functools import lru_cache
from pathlib import Path

# Configure model caches before loading ML libraries.
BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault(
    "HF_HOME",
    str(CACHE_DIR / "huggingface"),
)
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    str(CACHE_DIR / "sentence_transformers"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(CACHE_DIR),
)

import chromadb
import numpy as np

from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PDF_PATH = (
    BASE / "data" / "Talasalitaan Dataset (1).pdf"
)

EMBED_MODEL = "intfloat/multilingual-e5-small"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = 3

FALLBACK_ANSWER = (
    "Walang sapat na impormasyon sa ibinigay na dokumento."
)

SYSTEM_PROMPT = f"""
You are LalimSalita, a Filipino vocabulary learning assistant.

You help students understand difficult Filipino vocabulary
from the provided Talasalitaan PDF.

IMPORTANT RULES:

1. Answer in clear, student-friendly Filipino.

2. Use ONLY the retrieved document context.

3. Use the DEFINITION and CONTEXTUAL_MEANING fields
   whenever available.

4. Explain vocabulary in a way appropriate for
   Grade 8 students.

5. Cite the supporting PDF page number using:
   (Pahina X)

6. Do not invent definitions, examples, or citations.

7. Do not answer unrelated questions using outside knowledge.

8. If the context does not contain sufficient evidence,
   respond with exactly:

{FALLBACK_ANSWER}

9. A retrieved excerpt is not automatically relevant.
   Verify that it actually supports the question.

10. Keep answers understandable and reasonably concise.
"""


# --------------------------------------------------
# DOCUMENT FORMATTING
# --------------------------------------------------

FIELDS = (
    "WORD",
    "PART_OF_SPEECH",
    "DEFINITION",
    "CONTEXTUAL_MEANING",
    "STANZA",
    "SOURCE",
)

WORD_PATTERN = re.compile(
    r"(?im)^WORD:\s*([^\n]+)"
)


def format_vocabulary_text(raw_text):
    """
    Restore vocabulary field boundaries after
    extracting text from a PDF page.
    """

    text = re.sub(
        r"\s+",
        " ",
        raw_text or "",
    ).strip()

    text = re.sub(
        r"\bARALIN\s+(\d+)\b",
        r"\n\nARALIN \1\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bVOCABULARY\s+ENTRY\s+(\d+)\b",
        r"\n\nVOCABULARY ENTRY \1\n",
        text,
        flags=re.IGNORECASE,
    )

    for field in FIELDS:
        text = re.sub(
            rf"\s*\b({field})\s*:\s*",
            r"\n\1: ",
            text,
            flags=re.IGNORECASE,
        )

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    ).strip()


# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------

def process_pdf():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Vocabulary PDF not found: {PDF_PATH}"
        )

    reader = PdfReader(str(PDF_PATH))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []
    vocabulary_index = {}

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            print(
                f"Warning: page {page_number}: {exc}"
            )
            continue

        text = format_vocabulary_text(raw_text)

        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "id": f"p{page_number}_c{index}",
                "page": page_number,
                "text": chunk_text,
            })

        # Extract headwords from the normalized page.
        # Store the complete entry where possible.
        entry_sections = re.split(
            r"(?im)(?=^VOCABULARY ENTRY\s+\d+\s*$)",
            text,
        )

        for entry in entry_sections:
            match = WORD_PATTERN.search(entry)

            if not match:
                continue

            word = match.group(1).strip().casefold()

            if not word:
                continue

            vocabulary_index.setdefault(
                word,
                [],
            ).append({
                "page": page_number,
                "text": entry.strip(),
            })

    if not chunks:
        raise ValueError(
            "The PDF contains no extractable text."
        )

    print("========== LALIMSALITA ==========")
    print(f"PDF: {PDF_PATH.name}")
    print(f"Pages: {len(reader.pages)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vocabulary terms: {len(vocabulary_index)}")
    print("=================================")

    return chunks, vocabulary_index


# --------------------------------------------------
# EMBEDDING MODEL AND VECTOR DATABASE
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_pipeline():
    """
    Initialize the embedding model and build
    the in-memory vector index once per process.
    """

    print("Initializing LalimSalita RAG pipeline...")

    encoder = SentenceTransformer(
        EMBED_MODEL,
        cache_folder=str(
            CACHE_DIR / "sentence_transformers"
        ),
    )

    chunks, vocabulary_index = process_pdf()

    client = chromadb.EphemeralClient()

    collection = client.create_collection(
        name="lalimsalita_live",
        metadata={
            "hnsw:space": "cosine"
        },
    )

    # Insert embeddings in manageable batches.
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
            show_progress_bar=False,
        )

        collection.add(
            ids=[
                item["id"]
                for item in batch
            ],
            documents=[
                item["text"]
                for item in batch
            ],
            metadatas=[
                {"page": item["page"]}
                for item in batch
            ],
            embeddings=vectors.tolist(),
        )

    print(
        f"Indexed ChromaDB chunks: {collection.count()}"
    )

    return encoder, collection, vocabulary_index


# --------------------------------------------------
# EXACT VOCABULARY MATCHING
# --------------------------------------------------

def find_exact_vocabulary(question, vocabulary_index):
    """
    Find headwords explicitly mentioned in the
    student's question.
    """

    normalized_question = re.sub(
        r"\s+",
        " ",
        question.casefold(),
    ).strip()

    matches = []

    for word in sorted(
        vocabulary_index.keys(),
        key=len,
        reverse=True,
    ):
        pattern = (
            r"(?<!\w)"
            + re.escape(word)
            + r"(?!\w)"
        )

        if not re.search(pattern, normalized_question):
            continue

        for entry in vocabulary_index[word]:
            matches.append({
                "page": entry["page"],
                "text": entry["text"],
                "match_type": "exact",
                "matched_word": word,
            })

    return matches


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

def retrieve(question):
    encoder, collection, vocabulary_index = (
        load_pipeline()
    )

    query_vector = encoder.encode(
        ["query: " + question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # A. Semantic retrieval
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(
            TOP_K,
            collection.count(),
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    semantic_docs = []

    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        semantic_docs.append({
            "page": metadata["page"],
            "text": document,
            "distance": float(distance),
            "match_type": "semantic",
        })

    # B. Exact headword retrieval
    exact_docs = find_exact_vocabulary(
        question,
        vocabulary_index,
    )

    if exact_docs:
        exact_vectors = encoder.encode(
            [
                "passage: " + doc["text"]
                for doc in exact_docs
            ],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_array = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        for doc, vector in zip(
            exact_docs,
            exact_vectors,
        ):
            similarity = float(
                np.dot(query_array, vector)
            )

            doc["distance"] = max(
                0.0,
                1.0 - similarity,
            )

        exact_docs.sort(
            key=lambda item: item["distance"]
        )

    # C. Combine exact and semantic results
    combined = []
    seen = set()

    for doc in exact_docs + semantic_docs:
        key = (
            doc["page"],
            doc["text"].strip(),
        )

        if key in seen:
            continue

        seen.add(key)
        combined.append(doc)

        if len(combined) >= TOP_K:
            break

    for rank, doc in enumerate(combined, start=1):
        doc["rank"] = rank

    print("========== RETRIEVAL ==========")
    print(f"Question: {question}")

    for doc in combined:
        print(
            f"Rank {doc['rank']} | "
            f"Page {doc['page']} | "
            f"{doc['match_type']} | "
            f"Distance {doc['distance']:.4f}"
        )

    print("================================")

    return combined


# --------------------------------------------------
# GROQ RESPONSE GENERATION
# --------------------------------------------------

def rag_query(question, api_key):
    """
    Retrieve relevant vocabulary evidence and
    generate a grounded Filipino explanation.
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

    docs = retrieve(question)

    if not docs:
        return {
            "answer": FALLBACK_ANSWER,
            "retrieved": [],
        }

    context = "\n\n".join(
        (
            f"[Source: page {doc['page']}, "
            f"rank {doc['rank']}]\n"
            f"{doc['text']}"
        )
        for doc in docs
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
                    f"QUESTION: {question}"
                ),
            },
        ],
    )

    answer = (
        completion.choices[0].message.content or ""
    ).strip()

    if not answer:
        answer = FALLBACK_ANSWER

    # Do not expose irrelevant retrieval excerpts
    # when the model explicitly refuses the question.
    if answer.casefold() == FALLBACK_ANSWER.casefold():
        return {
            "answer": FALLBACK_ANSWER,
            "retrieved": [],
        }

    return {
        "answer": answer,
        "retrieved": docs,
    }
