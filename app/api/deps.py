# app/api/deps.py
# =============================================================================
from services.prediction_service import prediction_service

def get_prediction_service():
    """Dependency to get prediction service"""
    return prediction_ser