# =============================================================================
# app/core/config.py
# =============================================================================
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "BBB Prediction API"
    DESCRIPTION: str = "Blood-Brain Barrier Permeability Prediction using Machine Learning"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:3001",
        "https://localhost:3000",
    ]
    
    # Model settings
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "../../models/lgbm_model.pkl")
    DEFAULT_THRESHOLD: float = 0.5228
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_MOLECULES_PER_REQUEST: int = 1000
    
    # Feature settings
    SELECTED_FEATURES: List[str] = [
        'TopoPSA', 'nHBDon', 'PEOE_VSA10', 'ATSC1se', 'AATS5s', 'n4HRing', 'PEOE_VSA1', 
        'SdssC', 'AATS7dv', 'nAcid', 'AATS8s', 'MDEC-33', 'SlogP_VSA2', 'EState_VSA10', 
        'AATSC0v', 'ATSC1s', 'SLogP', 'ATS8dv', 'AATS7Z', 'VSA_EState3', 'SsssCH', 
        'C1SP2', 'IC1', 'AATS7d', 'Xch-5d', 'Kier2', 'AATS1i', 'Lipinski', 'VSA_EState5', 
        'GhoseFilter', 'AETA_eta_F', 'nN', 'VSA_EState2', 'SsNH2', 'MATS1c', 'nSZ', 
        'MIC3', 'NssSP', 'PEOE_VSA5', 'nRot', 'ATSC4c', 'GATS2i', 'PEOE_VSA13', 
        'BCUTi-1l', 'n7FA', 'HRing', 'ATSC2s', 'SlogP_VSA3', 'NdssC', 'BCUTs-1l', 
        'GATS8d', 'MDEC-22', 'ATSC7c', 'Xch-6d', 'VSA_EState7', 'C4SP3', 'GATS3d', 
        'IC2', 'GATS1c'
    ]
    
    # Supported file types
    SUPPORTED_FILE_TYPES: List[str] = ["csv", "sdf", "mol"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()