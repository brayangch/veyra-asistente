"""Inference NumPy del diminuto GRU propio de Veyra para planes de poca RAM."""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "modelo_generativo_publico.json"
_CACHE = None
TOKEN_RE = re.compile(r"[\wáéíóúñü]+|[^\w\s]", re.I)


def load_model():
    global _CACHE
    if _CACHE is None:
        data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if data.get("format") != "veyra-tiny-gru-numpy-v1" or not data.get("trained_from_scratch"):
            raise ValueError("El generador público no tiene el formato esperado.")
        for key in ("embedding", "weight_ih", "weight_hh", "bias_ih", "bias_hh", "head_weight", "head_bias"):
            data[key] = np.asarray(data[key], dtype=np.float32)
        _CACHE = data
    return _CACHE


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))


def _step(token_id: int, h: np.ndarray, m: dict) -> np.ndarray:
    x = m["embedding"][token_id]
    gi = m["weight_ih"] @ x + m["bias_ih"]
    gh = m["weight_hh"] @ h + m["bias_hh"]
    i_r, i_z, i_n = np.split(gi, 3)
    h_r, h_z, h_n = np.split(gh, 3)
    reset = _sigmoid(i_r + h_r)
    update = _sigmoid(i_z + h_z)
    candidate = np.tanh(i_n + reset * h_n)
    return candidate + update * (h - candidate)


def generate(prompt: str, max_tokens: int = 60) -> str:
    m = load_model()
    vocab = m["vocabulary"]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    parts = TOKEN_RE.findall(prompt.lower())
    ids = [word_to_id[p] for p in parts if p in word_to_id][-40:]
    hidden = np.zeros(m["config"]["hidden"], dtype=np.float32)
    ids = [word_to_id["<inicio>"]] + ids
    for token_id in ids:
        hidden = _step(token_id, hidden, m)
    rng = random.Random()
    output = []
    pad_id, start_id, end_id = (word_to_id[k] for k in ("<relleno>", "<inicio>", "<fin>"))
    for _ in range(max(1, min(int(max_tokens), 100))):
        scores = (m["head_weight"] @ hidden + m["head_bias"]) / 0.8
        scores[pad_id] = -np.inf
        scores[start_id] = -np.inf
        ids = np.argpartition(scores, -min(8, len(scores) - 2))[-min(8, len(scores) - 2):]
        ids = ids[np.argsort(scores[ids])[::-1]]
        values = scores[ids]
        values -= np.max(values)
        probs = np.exp(values)
        probs /= probs.sum()
        token_id = int(rng.choices(ids.tolist(), weights=probs.tolist(), k=1)[0])
        if token_id == end_id:
            break
        output.append(vocab[token_id])
        hidden = _step(token_id, hidden, m)
    text = ""
    punctuation = {".", ",", "?", "!", ":", ";"}
    for token in output:
        text += (token if token in punctuation else token + " ")
    return text.strip() or "(El modelo no generó texto para esta semilla.)"
