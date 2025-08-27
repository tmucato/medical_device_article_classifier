from flask import Flask, jsonify, request
import hashlib
import os
from typing import List, Union

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


DEFAULT_VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))


def _require_numpy() -> None:
    if _np is None:
        raise RuntimeError(
            "numpy is required but not installed. Ensure it is in requirements.txt"
        )


def _text_to_vector(text: str, size: int = DEFAULT_VECTOR_SIZE) -> List[float]:
    """
    Deterministically convert text to a pseudo-embedding of fixed size using a
    hash-based PRNG. This is a lightweight stand-in for a real model so the
    service works out of the box without heavy dependencies.
    """
    _require_numpy()
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = _np.random.RandomState(seed)
    vec = rng.normal(loc=0.0, scale=1.0, size=size)
    # L2 normalize
    norm = _np.linalg.norm(vec) or 1.0
    vec = (vec / norm).astype(float)
    return vec.tolist()


def _ensure_texts(payload: dict) -> List[str]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    if "texts" in payload:
        texts = payload["texts"]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("'texts' must be a list of strings")
        return texts
    if "text" in payload:
        text = payload["text"]
        if not isinstance(text, str):
            raise ValueError("'text' must be a string")
        return [text]
    raise ValueError("Provide 'text' (string) or 'texts' (list of strings)")


app = Flask(__name__)


@app.get("/health")
def health() -> Union[tuple, tuple]:
    return jsonify({"status": "ok", "vector_size": DEFAULT_VECTOR_SIZE}), 200


@app.post("/embed")
def embed() -> Union[tuple, tuple]:
    try:
        payload = request.get_json(silent=True) or {}
        texts = _ensure_texts(payload)
        vectors = [_text_to_vector(t, DEFAULT_VECTOR_SIZE) for t in texts]
        if "text" in payload and "texts" not in payload:
            return jsonify({"embedding": vectors[0]}), 200
        return jsonify({"embeddings": vectors}), 200
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)


