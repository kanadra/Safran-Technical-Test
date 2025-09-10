from __future__ import annotations

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

from .logging_conf import setup_logging
from .database import (
    init_db, create_user, get_user_by_email,
    create_prediction, list_predictions, get_prediction, get_stats
)
from .auth import (
    hash_password, verify_password,
    create_access_token, decode_token, InvalidTokenError
)
from .inference import predict
from .schemas import UserCreate, Token, PredictIn, PredictOut, PredictionItem

# --- Setup ---

logger = setup_logging()
init_db()

app = FastAPI(
    title="Practical AI API",
    version="1.2.0",
    description=(
        "FastAPI service with JWT auth, SQLite persistence, multi-model "
        "inference (ONNX-ready), input validation, logging and stats."
    ),
)

# Allow all origins (dev/testing only; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reusable HTTP bearer scheme (adds Authorization header to OpenAPI)
bearer_scheme = HTTPBearer(auto_error=False)

# --- Dependency: current user ---

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> dict:
    """Extract and return user from Authorization header (JWT Bearer)."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    token = credentials.credentials
    # Some clients may prepend "Bearer " redundantly → strip if present
    if token.lower().startswith("bearer "):
        token = token[7:]

    try:
        payload = decode_token(token)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    email = payload.get("sub")
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

# --- Endpoints ---

@app.post("/api/register", response_model=Token, status_code=201)
def register(payload: UserCreate) -> Token:
    """Register a new user and return an access token."""
    if get_user_by_email(payload.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    password_hash = hash_password(payload.password)
    user_id = create_user(payload.email, password_hash)
    token = create_access_token(payload.email)
    logger.info("Created user %s with ID %s", payload.email, user_id)
    return Token(access_token=token)

@app.post("/api/login", response_model=Token)
def login(payload: UserCreate) -> Token:
    """Authenticate a user and return an access token."""
    user = get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(payload.email)
    logger.info("User %s logged in", payload.email)
    return Token(access_token=token)

@app.post("/api/predictions", response_model=PredictOut)
def create_prediction_endpoint(payload: PredictIn, user: dict = Depends(get_current_user)) -> PredictOut:
    """Run inference on text and store result."""
    result = predict(payload.text, payload.model_version)
    pred_id = create_prediction(
        user_id=user["id"],
        model_version=result.get("model_version", payload.model_version),
        input_data={"text": payload.text},
        output_data={"label": result["label"], "score": result["score"]},
    )
    logger.info(
        "Stored prediction %s for user %s using %s",
        pred_id, user["email"], result["model_version"]
    )
    return PredictOut(
        id=pred_id,
        model_version=result["model_version"],
        label=result["label"],
        score=result["score"],
    )

@app.get("/api/predictions", response_model=List[PredictionItem])
def list_predictions_endpoint(user: dict = Depends(get_current_user)) -> List[PredictionItem]:
    """Return all predictions for the current user."""
    records = list_predictions(user["id"])
    # Deserialize stored JSON fields
    for rec in records:
        try:
            rec["input_data"] = json.loads(rec["input_data"])
            rec["output_data"] = json.loads(rec["output_data"])
        except Exception:
            pass
    return [PredictionItem(**rec) for rec in records]

@app.get("/api/predictions/{prediction_id}", response_model=PredictionItem)
def get_prediction_endpoint(prediction_id: int, user: dict = Depends(get_current_user)) -> PredictionItem:
    """Retrieve a single prediction record by ID."""
    rec = get_prediction(user["id"], prediction_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    try:
        rec["input_data"] = json.loads(rec["input_data"])
        rec["output_data"] = json.loads(rec["output_data"])
    except Exception:
        pass
    return PredictionItem(**rec)

@app.get("/api/stats")
def stats_endpoint(user: dict = Depends(get_current_user)) -> dict:
    """Return summary metrics for the current user."""
    return get_stats(user["id"])

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}

# Import json at the end to avoid circular import
import json  # noqa: E402
