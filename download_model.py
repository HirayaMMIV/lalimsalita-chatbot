
import os
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "onnx_model"

MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "hirayammiv/lalimsalita-e5-onnx"
)

print("Downloading LalimSalita ONNX model...")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=str(MODEL_DIR),
    allow_patterns=[
        "model_int8.onnx",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)

required_files = [
    "model_int8.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

for filename in required_files:
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

print("Model downloaded successfully!")
