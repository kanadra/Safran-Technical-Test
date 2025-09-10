# Practical Sentiment Analysis API (SQLite + FastAPI)

This repository contains a complete example of a RESTful API for
**sentiment analysis** powered by FastAPI.  
It demonstrates how to build a self-contained service with:

- User authentication (JWT-style tokens)
- SQLite persistence (no external database)
- Multiple model versions (rule-based fallback + optional ONNX)
- Request validation with Pydantic
- Logging, history, and simple statistics
- Swagger/OpenAPI documentation for testing

Everything runs with only FastAPI, Pydantic, and Python's standard
library — no external ORMs or cryptography packages are required.

---

## Features

- **SQLite database** with a simple schema for users and predictions.  
- **Authentication** using HMAC-signed JWT-like tokens (HS256).  
- **Input validation** with Pydantic:  
  - Emails must be ≥ 3 characters  
  - Passwords must be ≥ 6 characters  
  - Prediction text must be non-empty and ≤ 5000 characters  
- **Logging** to both console and rotating log files (`logs/app.log`).  
- **Multiple model versions**:  
  - `v1` → default model  
  - `v2` → alternate version (same API, can point to a different ONNX)  
  - If no ONNX model is present, the API falls back to a simple deterministic rule-based classifier.  
- **/stats endpoint** to see totals, class distribution, and usage by model version.  
- **Swagger UI** (`/docs`) for interactive testing.

---

## About the sentiment models

This project is preconfigured with a lightweight **ONNX sentiment analysis model**:

- **Slim-Sentiment ONNX**  
  - Based on a quantized transformer model for text classification  
  - Optimized for CPU (int4 quantization)  
  - Trained to classify text into **positive** vs **negative** sentiment  

You can replace the included model with any compatible ONNX model
from [Hugging Face](https://huggingface.co/models?library=onnx&pipeline_tag=text-classification).  
For example:
- `MiniLMv2 GoEmotions V2 ONNX` (emotion classification)  
- `distilbert-base-uncased-finetuned-sst-2-english` (sentiment classification)  

To swap models, download the `.onnx` file and set:

```bash
export MODEL_V1_PATH=/path/to/model.onnx
