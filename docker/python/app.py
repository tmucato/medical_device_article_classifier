from flask import Flask, jsonify, request
import os
import requests
from typing import Union


BIOBERT_URL = os.getenv("BIOBERT_INFERENCE_URL", "http://localhost:5000")

app = Flask(__name__)


@app.get("/health")
def health() -> Union[tuple, tuple]:
    try:
        r = requests.get(f"{BIOBERT_URL}/health", timeout=5)
        upstream = r.json() if r.ok else {"status": "unreachable"}
    except Exception:
        upstream = {"status": "unreachable"}
    return jsonify({"status": "ok", "biobert": upstream}), 200


@app.post("/embed")
def embed() -> Union[tuple, tuple]:
    payload = request.get_json(silent=True) or {}
    try:
        r = requests.post(f"{BIOBERT_URL}/embed", json=payload, timeout=30)
        return jsonify(r.json()), r.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)


