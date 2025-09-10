# Practical Sentiment Analysis API (SQLite + FastAPI)

This repository contains a complete example of a RESTful API for **sentiment analysis** powered by FastAPI.  
It demonstrates how to build a self‑contained service with:

- **User authentication** using HMAC‑signed JWT tokens (HS256)
- **SQLite persistence** (no external database required)
- **Multiple model versions** (rule‑based fallback or optional ONNX models)
- **Request validation** with Pydantic
- **Logging**, history and simple statistics
- **Swagger/OpenAPI documentation** for interactive testing

Everything runs with only FastAPI, Pydantic and Python’s standard library — no external ORMs or cryptography packages are required.

---

## Features

- **SQLite database** with a simple schema for users and predictions.
- **Authentication** using HMAC‑signed JWT‑like tokens (HS256).
- **Input validation** with Pydantic:
  - Emails must be ≥ 3 characters
  - Passwords must be ≥ 6 characters
  - Prediction text must be non‑empty and ≤ 5000 characters
- **Logging** to both console and rotating log files (`logs/app.log`).
- **Multiple model versions**:
  - `v1` → default model
  - `v2` → alternate version (same API, can point to a different ONNX model)
  - If no ONNX model is present, the API falls back to a simple deterministic rule‑based classifier.
- **/stats endpoint** to see totals, class distribution and usage by model version.
- **Swagger UI** (`/docs`) for interactive testing.

---

## About the sentiment models

This project is preconfigured with a lightweight **ONNX sentiment analysis model**:

- **Slim‑Sentiment ONNX**
  - Based on a quantized transformer model for text classification
  - Optimized for CPU (int4 quantization)
  - Trained to classify text into **positive** vs **negative** sentiment

You can replace the included model with any compatible ONNX model from the [Hugging Face model hub](https://huggingface.co/models?library=onnx&pipeline_tag=text-classification). For example:

- `MiniLMv2 GoEmotions V2 ONNX` (emotion classification)
- `distilbert‑base‑uncased‑finetuned‑sst‑2‑english` (sentiment classification)

To swap models, download the `.onnx` file and set the `MODEL_V1_PATH` environment variable. For example:

```bash
export MODEL_V1_PATH=/path/to/model.onnx
```

---

## Running the API

### Install dependencies

To set up the environment, run:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install fastapi uvicorn
pip install onnxruntime transformers   # optional, for real ONNX models
```

### Start the server

Launch the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

By default, the server looks for models in `models/model_v1.onnx` and `models/model_v2.onnx`. If these files are not found, it will use the built‑in fallback (a simple rule‑based classifier).
I recommend using distilbert-base-uncased-finetuned-sst-2-english, the ONNX file is located [here](https://huggingface.co/models?library=onnx&pipeline_tag=text-classification).

Place this in the models folder and rename to model_v1.onnx

### Open Swagger UI

Once the server is running, navigate to the documentation at:

```
http://127.0.0.1:8000/docs
```

### Register and log in

To use the API you need to register a user, then log in to obtain an access token. Use the `/api/register` endpoint to create an account and `/api/login` to obtain a JWT token. In the Swagger UI you can click the **Authorize** button to paste your token and authenticate future requests.

### Make predictions and view stats

- **/api/predictions** (POST): submit text to classify
- **/api/predictions** (GET): list all predictions for the authenticated user
- **/api/predictions/{id}** (GET): fetch a single prediction by ID
- **/api/stats** (GET): see total predictions, counts by sentiment and counts by model version

## Run local tests

Local tests are provided to verify functionality:

```bash
python tests_local.py
```

This script uses FastAPI’s `TestClient` to register, log in, make predictions and validate the stats output.

If you wish to reset the database, delete or rename `app.db` before running the server.

---

## Notes

- The API defaults to a mock sentiment classifier if no ONNX model is available.
- To test with a real model, install `onnxruntime`, place an ONNX sentiment model under `models/`, and restart the server.

---

## Example prediction output

A sample JSON response from the prediction endpoint looks like this:

```json
{
  "id": 1,
  "model_version": "v1",
  "label": "POSITIVE",
  "score": 0.92
}
```

---

## Example usage with curl

### Register a user

```bash
curl -X POST http://127.0.0.1:8000/api/register \ 
     -H "Content-Type: application/json" \ 
     -d '{"email":"test@example.com","password":"secret123"}'
```

### Log in and capture the token

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/login \ 
     -H "Content-Type: application/json" \ 
     -d '{"email":"test@example.com","password":"secret123"}' | jq -r .access_token)
```

### Make a prediction

```bash
curl -X POST http://127.0.0.1:8000/api/predictions \ 
     -H "Content-Type: application/json" \ 
     -H "Authorization: Bearer $TOKEN" \ 
     -d '{"text":"I absolutely loved this!", "model_version":"v1"}'
```

The above commands demonstrate how to register, authenticate, and perform a sentiment prediction using the API from the command line.