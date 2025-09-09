# app/api/v1/api.py
# =============================================================================
from fastapi import APIRouter

from api.v1.endpoints import predictions

api_router = APIRouter()
api_router.include_router(predictions.Router, prefix="/predictions", tags=["predictions"])

