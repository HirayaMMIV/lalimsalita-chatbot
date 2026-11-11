
"""LalimSalita lightweight ONNX retrieval backend (Render-friendly).
Keep the quantized model and embeddings generated with that exact model together.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import json
import logging
import re
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from dotenv import load_dotenv
from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')
DATA_DIR = BASE / 'retrieval_data'
MODEL_DIR = BASE / 'onnx_model'
MODEL_PATH = MODEL_DIR / 'model_int8.onnx'
TOKENIZER_PATH = MODEL_DIR / 'tokenizer.json'
MODEL_REPO = os.getenv('HF_MODEL_REPO', 'hirayammiv/lalimsalita-e5-onnx')
TOP_K = 3
MAX_QUERY_TOKENS = 128
FALLBACK = 'Walang sapat na impormasyon sa ibinigay na dokumento.'
FALLBACK_ANSWER = FALLBACK
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
logger = logging.getLogger('lalimsalita.backend')
logging.basicConfig(level=logging.INFO)
_LOCK = threading.Lock()
_SESSION = None
_TOKENIZER = None


def load_retrieval_data():
    with (DATA_DIR / 'chunks.json').open(encoding='utf-8') as f:
        chunks = json.load(f)
    if isinstance(chunks, dict):
        chunks = chunks.get('chunks', chunks.get('documents', []))
    embeddings = np.load(DATA_DIR / 'embeddings.npy', allow_pickle=False).astype(np.float32)
    if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
        raise ValueError('chunks.json and embeddings.npy do not have matching rows')
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    logger.info('Loaded %d precomputed chunks.', len(chunks))
    return chunks, embeddings


CHUNKS, EMBEDDINGS = load_retrieval_data()


def ensure_model_files():
    missing = [f for f in ('model_int8.onnx', 'tokenizer.json') if not (MODEL_DIR / f).exists()]
    if not missing:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    for filename in missing:
        logger.info('Downloading %s from Hugging Face...', filename)
        hf_hub_download(repo_id=MODEL_REPO, filename=filename,
                        local_dir=str(MODEL_DIR), token=os.getenv('HF_TOKEN') or None)


def get_model():
    global _SESSION, _TOKENIZER
    if _SESSION is not None:
        return _SESSION, _TOKENIZER
    with _LOCK:
        if _SESSION is not None:
            return _SESSION, _TOKENIZER
        ensure_model_files()
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        tokenizer.enable_truncation(max_length=MAX_QUERY_TOKENS)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(MODEL_PATH), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        _TOKENIZER, _SESSION = tokenizer, session
        logger.info('Quantized ONNX model initialized.')
    return _SESSION, _TOKENIZER


def get_query_embedding(question):
    session, tokenizer = get_model()
    encoded = tokenizer.encode('query: ' + str(question).strip(), add_special_tokens=True)
    ids = np.asarray([encoded.ids], dtype=np.int64)
    mask = np.asarray([encoded.attention_mask], dtype=np.int64)
    inputs = {'input_ids': ids, 'attention_mask': mask}
    if 'token_type_ids' in {x.name for x in session.get_inputs()}:
        inputs['token_type_ids'] = np.zeros_like(ids)
    output = np.asarray(session.run(None, inputs)[0], dtype=np.float32)
    if output.ndim == 3:
        expanded = mask[:, :output.shape[1], None].astype(np.float32)
        vector = ((output * expanded).sum(axis=1) /
                  np.maximum(expanded.sum(axis=1), 1e-12))[0]
    elif output.ndim == 2 and output.shape[0] == 1:
        vector = output[0]
    else:
        raise ValueError(f'Unexpected ONNX output: {output.shape}')
    if len(vector) != EMBEDDINGS.shape[1]:
        raise ValueError('Embedding dimension mismatch; rebuild embeddings.npy with this model')
    return (vector / max(np.linalg.norm(vector), 1e-12)).astype(np.float32)


def normalize_text(value):
    return ' '.join(re.sub(r'[^\w\s]', ' ', str(value).casefold()).split())


def extract_target_word(question):
    question = normalize_text(question)
    patterns = [
        r'(?:kahulugan|ibig sabihin|depinisyon)\s+(?:ng|ni|sa)\s+(.+)',
        r'(?:ano\s+ang)\s+(.+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            word = match.group(1).strip()
            word = re.sub(r'^(?:salitang|salita)\s+', '', word)
            return word
    return question


def extract_vocabulary_word(text):
    match = re.search(r'\bWORD\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if match:
        return normalize_text(match.group(1))
    return ''


def get_chunk_id(chunk, index):
    return chunk.get('id', chunk.get('chunk_id', index))


def get_chunk_page(chunk):
    return chunk.get('page', chunk.get('page_number', 0))


def get_chunk_text(chunk):
    return str(chunk.get('text', chunk.get('page_content', '')))


def retrieve(question, top_k=TOP_K):
    query_vector = get_query_embedding(question)
    similarities = EMBEDDINGS @ query_vector
    target = extract_target_word(question)

    scored = []
    for index, chunk in enumerate(CHUNKS):
        text = get_chunk_text(chunk)
        vocabulary_word = extract_vocabulary_word(text)
        exact_match = bool(target and vocabulary_word == target)
        scored.append((index, float(similarities[index]), exact_match))

    scored.sort(key=lambda row: (not row[2], -row[1]))
    results = []
    for rank, (index, similarity, exact_match) in enumerate(scored[:top_k], 1):
        chunk = CHUNKS[index]
        results.append({
            'chunk_id': get_chunk_id(chunk, index),
            'page': get_chunk_page(chunk),
            'text': get_chunk_text(chunk),
            'similarity': similarity,
            'distance': 1.0 - similarity,
            'match_type': 'exact vocabulary' if exact_match else 'semantic',
            'rank': rank,
        })
    logger.info('Retrieved %d chunks for query.', len(results))
    return results


def build_context(retrieved):
    parts = []
    for item in retrieved:
        parts.append(
            f"[Page {item['page']} | Rank {item['rank']}]\n{item['text']}"
        )
    return '\n\n'.join(parts)


def generate_answer(question, retrieved, api_key=None):
    api_key = api_key or os.getenv('GROQ_API_KEY')
    if not api_key:
        raise RuntimeError('GROQ_API_KEY is missing. Add it to Render Environment.')

    context = build_context(retrieved)
    system_prompt = (
        'Ikaw si LalimSalita, isang Filipino vocabulary learning companion. '
        'Sumagot lamang batay sa ibinigay na konteksto mula sa Talasalitaan. '
        'Ipaliwanag nang malinaw ang kahulugan at kontekstuwal na gamit '
        'ng salitang tinatanong. Kung walang sapat na impormasyon, '
        f'sabihin nang eksakto: {FALLBACK}'
    )
    payload = {
        'model': GROQ_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': f'KONTEKSTO:\n{context}\n\nTANONG:\n{question}',
            },
        ],
        'temperature': 0.2,
        'max_tokens': 350,
    }
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['message']['content'].strip()


def rag_query(question, api_key=None):
    retrieved = retrieve(question, top_k=TOP_K)
    answer = generate_answer(question, retrieved, api_key=api_key)
    return {
        'answer': answer,
        'retrieved': retrieved,
    }
