from __future__ import annotations

import os
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- ONNX availability ---
try:
    import onnxruntime as ort  # type: ignore
    import numpy as np
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except Exception as e:
    ONNX_AVAILABLE = False
    logger.info("onnxruntime or dependencies not available; using mock inference: %s", e)

# --- Model paths ---
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODEL_DIR = os.path.join(_BASE_DIR, "models")

MODEL_V1_PATH = os.getenv("MODEL_V1_PATH", os.path.join(_DEFAULT_MODEL_DIR, "model_v1.onnx"))
MODEL_V2_PATH = os.getenv("MODEL_V2_PATH", os.path.join(_DEFAULT_MODEL_DIR, "model_v2.onnx"))

# Keep track of loaded sessions + tokenizers
_sessions: Dict[str, Any] = {"v1": None, "v2": None}
_tokenizers: Dict[str, Any] = {"v1": None, "v2": None}


# --- Helpers ---
def _load_onnx_session(version: str) -> None:
    """Load an ONNX model + tokenizer for the given version if available."""
    path = MODEL_V1_PATH if version == "v1" else MODEL_V2_PATH
    if ONNX_AVAILABLE and os.path.exists(path):
        try:
            _sessions[version] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            logger.info("Loaded ONNX model %s from %s", version, path)

            try:
                tok = AutoTokenizer.from_pretrained(_DEFAULT_MODEL_DIR)

                # Ensure a pad token exists (GPT-style models often don’t have one)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                    logger.info("Set pad_token to eos_token for %s", version)

                _tokenizers[version] = tok
                logger.info("Loaded tokenizer for %s", version)

            except Exception as tok_exc:
                logger.warning("Tokenizer not available for %s: %s", version, tok_exc)
                _tokenizers[version] = None

        except Exception as exc:
            logger.exception("Failed to load ONNX model %s: %s", version, exc)
            _sessions[version] = None
    else:
        _sessions[version] = None


# --- Main inference API ---
def predict(text: str, version: str = "v1") -> Dict[str, Any]:
    """Run inference on text using model `version` ("v1" or "v2")."""
    start_time = time.time()
    if version not in ("v1", "v2"):
        version = "v1"

    # Load session if not already
    if _sessions[version] is None:
        _load_onnx_session(version)

    # --- Mock fallback ---
    if _sessions[version] is None:
        length = len(text)
        label = "NEGATIVE" if (length % (2 if version == "v1" else 3)) == 0 else "POSITIVE"
        score = (length * (1 if version == "v1" else 7)) % 100 / 100.0
        elapsed = (time.time() - start_time) * 1000.0
        return {"label": label, "score": score, "model_version": version, "elapsed_ms": elapsed}

    # --- ONNX path ---
    session = _sessions[version]
    try:
        input_names = [i.name for i in session.get_inputs()]

        if "input_ids" in input_names and _tokenizers[version]:
            encoded = _tokenizers[version](
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=32,
            )
            inputs = {k: v.astype(np.int64) for k, v in encoded.items() if k in input_names}

            # Fill in any missing keys (e.g., past_key_values.*) with zeros
            for inp in input_names:
                if inp not in inputs:
                    shape = [s if isinstance(s, int) else 1 for s in session.get_inputs()[input_names.index(inp)].shape]
                    inputs[inp] = np.zeros(shape, dtype=np.float32)

            outputs = session.run(None, inputs)
            val = float(outputs[0].ravel()[0])
            label = "NEGATIVE" if val >= 0 else "POSITIVE"
            score = 1 / (1 + np.exp(-abs(val)))

        else:
            # Toy numeric input model
            input_name = session.get_inputs()[0].name
            arr = np.array([[len(text)]], dtype=np.float32)
            outputs = session.run(None, {input_name: arr})
            val = float(outputs[0].ravel()[0])
            label = "NEGATIVE" if val >= 0 else "POSITIVE"
            score = 1 / (1 + np.exp(-abs(val)))

    except Exception as exc:
        logger.exception("ONNX inference failed for %s: %s", version, exc)
        # Fallback: deterministic mock
        length = len(text)
        label = "NEGATIVE" if length % 2 == 0 else "POSITIVE"
        score = (length % 100) / 100.0

    elapsed = (time.time() - start_time) * 1000.0
    return {"label": label, "score": score, "model_version": version, "elapsed_ms": elapsed}
