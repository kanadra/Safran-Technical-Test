from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Any, Dict

# --- Auth schemas ---

class UserCreate(BaseModel):
    """Schema for user registration and login requests."""
    email: str = Field(
        ..., min_length=3, max_length=320,
        description="Email address (unique identifier)."
    )
    password: str = Field(
        ..., min_length=6, max_length=128,
        description="Password (min 6 characters)."
    )

class Token(BaseModel):
    """Response schema for authentication endpoints."""
    access_token: str
    token_type: str = "bearer"

# --- Prediction schemas ---

class PredictIn(BaseModel):
    """Request schema for prediction endpoint."""
    text: str = Field(
        ..., min_length=1, max_length=5000,
        description="Input text to classify (1–5000 chars)."
    )
    model_version: Literal["v1", "v2"] = Field(
        "v1",
        description="'v1' base model or 'v2' fine-tuned variant."
    )

class PredictOut(BaseModel):
    """Response schema for prediction endpoint."""
    id: int
    model_version: str
    label: str
    score: float

class PredictionItem(BaseModel):
    """Schema for stored prediction records."""
    id: int
    model_version: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    created_at: str

    class Config:
        from_attributes = True  # allow mapping from SQLite row dicts