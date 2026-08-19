<<<<<<< HEAD
"""Run the BBB prediction API.

The service is mounted behind the Replit ``/api`` proxy prefix.  The FastAPI
application itself accepts both direct routes and prefixed routes so it also
works when run locally.
"""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "bbb_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
=======
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
import io
import logging
from datetime import datetime

load_dotenv()

from bbb_prediction import (
    predict_single_molecule,
    predict_multiple_molecules,
    predict_from_file,
    get_model_info,
    test_external_molecules,
    debug_sdf_content,
    validate_smiles,
    clean_smiles,
    convert_numpy_types,
    SELECTED_FEATURES,
    MORDRED_AVAILABLE,
    model
)

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_PATH = os.getenv("MODEL_PATH", "lgbm_model.pkl")
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", 0.5228))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 100))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 1000))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BBB Permeability Prediction API",
    description="Blood-Brain Barrier permeability prediction using LightGBM and Mordred descriptors",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,  # Disable docs in production
    redoc_url="/redoc" if DEBUG else None,
    debug=DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class SingleMoleculeRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    smiles: str = Field(..., description="SMILES string of the molecule")
    name: Optional[str] = Field("Unknown", description="Name of the molecule")
    threshold: Optional[float] = Field(MODEL_THRESHOLD, description="Classification threshold", ge=0.0, le=1.0)
    
    @validator('smiles')
    def validate_smiles_string(cls, v):
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        return v.strip()

class BatchMoleculeRequest(BaseModel):
    molecules: List[Dict[str, str]] = Field(..., description="List of molecules with SMILES and names")
    threshold: Optional[float] = Field(MODEL_THRESHOLD, description="Classification threshold", ge=0.0, le=1.0)
    
    @validator('molecules')
    def validate_molecules_list(cls, v):
        if not v:
            raise ValueError("Molecules list cannot be empty")
        
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum limit of {MAX_BATCH_SIZE}")
        
        for i, mol in enumerate(v):
            if 'smiles' not in mol:
                raise ValueError(f"Molecule at index {i} is missing 'smiles' field")
            if not mol['smiles'] or not mol['smiles'].strip():
                raise ValueError(f"Molecule at index {i} has empty SMILES string")
        
        return v

class FileUploadMetadata(BaseModel):
    smiles_column: Optional[str] = Field(None, description="Name of SMILES column (CSV only)")
    name_column: Optional[str] = Field(None, description="Name of compound name column (CSV only)")
    threshold: Optional[float] = Field(MODEL_THRESHOLD, description="Classification threshold", ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    smiles: str
    name: str
    status: str
    prediction: Optional[str]
    probability_bbb_negative: Optional[float]
    probability_bbb_positive: Optional[float]
    confidence: Optional[float]
    threshold_used: Optional[float]
    interpretation: str

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_type: Optional[str]
    feature_count: Optional[int]
    features: Optional[List[str]]
    descriptor_library: Optional[str]
    mordred_available: bool
    model_classes: Optional[List[str]]
    default_threshold: Optional[float]
    supported_file_types: Optional[List[str]]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BBB Permeability Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if DEBUG else "Documentation disabled in production",
        "environment": "development" if DEBUG else "production"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "mordred_available": MORDRED_AVAILABLE,
        "environment": "development" if DEBUG else "production",
        "model_path": MODEL_PATH,
        "default_threshold": MODEL_THRESHOLD
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_information():
    """Get information about the loaded BBB prediction model"""
    try:
        info = get_model_info()
        return JSONResponse(content=convert_numpy_types(info))
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model information: {str(e)}")

@app.post("/predict/single")
async def predict_single(request: SingleMoleculeRequest):
    """Predict BBB permeability for a single molecule"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="BBB prediction model not loaded")
        
        if not MORDRED_AVAILABLE:
            raise HTTPException(status_code=503, detail="Mordred library not available")
        
        cleaned_smiles = clean_smiles(request.smiles)
        if not validate_smiles(cleaned_smiles):
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        result = predict_single_molecule(
            smiles=cleaned_smiles,
            name=request.name,
            threshold=request.threshold
        )
        
        return JSONResponse(content=convert_numpy_types(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchMoleculeRequest):
    """Predict BBB permeability for multiple molecules"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="BBB prediction model not loaded")
        
        if not MORDRED_AVAILABLE:
            raise HTTPException(status_code=503, detail="Mordred library not available")
        
        validated_molecules = []
        for i, mol in enumerate(request.molecules):
            cleaned_smiles = clean_smiles(mol['smiles'])
            if not validate_smiles(cleaned_smiles):
                logger.warning(f"Invalid SMILES at index {i}: {mol['smiles']}")
                continue
                
            validated_molecules.append({
                'smiles': cleaned_smiles,
                'name': mol.get('name', f'Compound_{i+1}')
            })
        
        if not validated_molecules:
            raise HTTPException(status_code=400, detail="No valid molecules found in request")
        
        predictions = predict_multiple_molecules(validated_molecules, request.threshold)
        
        successful = [p for p in predictions if p['status'] == 'Success']
        bbb_positive = [p for p in successful if p['prediction'] == 'BBB+']
        bbb_negative = [p for p in successful if p['prediction'] == 'BBB-']
        high_conf = [p for p in successful if p['confidence'] >= 0.8]
        medium_conf = [p for p in successful if 0.6 <= p['confidence'] < 0.8]
        low_conf = [p for p in successful if p['confidence'] < 0.6]
        
        summary = {
            'total_molecules': len(predictions),
            'successful_predictions': len(successful),
            'failed_predictions': len(predictions) - len(successful),
            'bbb_positive': len(bbb_positive),
            'bbb_negative': len(bbb_negative),
            'high_confidence': len(high_conf),
            'medium_confidence': len(medium_conf),
            'low_confidence': len(low_conf),
            'success_rate': len(successful) / len(predictions) * 100 if predictions else 0,
            'bbb_positive_rate': len(bbb_positive) / len(successful) * 100 if successful else 0,
            'threshold_used': request.threshold
        }
        
        result = {
            'predictions': predictions,
            'summary': summary
        }
        
        return JSONResponse(content=convert_numpy_types(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/file")
async def predict_from_uploaded_file(
    file: UploadFile = File(..., description="Chemical file (CSV, SDF, MOL)"),
    threshold: float = Query(MODEL_THRESHOLD, description="Classification threshold", ge=0.0, le=1.0),
    smiles_column: Optional[str] = Query(None, description="SMILES column name (CSV only)"),
    name_column: Optional[str] = Query(None, description="Name column name (CSV only)")
):
    """Predict BBB permeability from uploaded file"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="BBB prediction model not loaded")
        
        if not MORDRED_AVAILABLE:
            raise HTTPException(status_code=503, detail="Mordred library not available")
        
        filename = file.filename.lower() if file.filename else ""
        if filename.endswith('.csv'):
            file_type = 'csv'
        elif filename.endswith('.sdf'):
            file_type = 'sdf'
        elif filename.endswith('.mol'):
            file_type = 'mol'
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Supported formats: .csv, .sdf, .mol"
            )
        
        content = await file.read()
        
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)"
            )
        
        try:
            file_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                file_content = content.decode('latin-1')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file content")
        
        kwargs = {}
        if smiles_column:
            kwargs['smiles_column'] = smiles_column
        if name_column:
            kwargs['name_column'] = name_column
        if file_type == 'mol':
            kwargs['name'] = filename.replace('.mol', '') or 'Unknown'
        
        result = predict_from_file(file_content, file_type, threshold, **kwargs)
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'File processing failed'))
        
        return JSONResponse(content=convert_numpy_types(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/validate/smiles")
async def validate_smiles_string(smiles: str = Form(...)):
    """Validate a SMILES string"""
    try:
        cleaned_smiles = clean_smiles(smiles)
        is_valid = validate_smiles(cleaned_smiles)
        
        return {
            "original_smiles": smiles,
            "cleaned_smiles": cleaned_smiles,
            "is_valid": is_valid,
            "message": "Valid SMILES structure" if is_valid else "Invalid SMILES structure"
        }
        
    except Exception as e:
        logger.error(f"Error validating SMILES: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SMILES validation failed: {str(e)}")

@app.post("/debug/sdf")
async def debug_sdf_file(file: UploadFile = File(..., description="SDF file to debug")):
    """Debug SDF file structure and parsing"""
    if not DEBUG:
        raise HTTPException(status_code=404, detail="Debug endpoints not available in production")
    
    try:
        if not file.filename or not file.filename.lower().endswith('.sdf'):
            raise HTTPException(status_code=400, detail="File must be an SDF file (.sdf extension)")
        
        content = await file.read()
        try:
            file_content = content.decode('utf-8')
        except UnicodeDecodeError:
            file_content = content.decode('latin-1')
        
        debug_info = debug_sdf_content(file_content)
        
        return JSONResponse(content=convert_numpy_types(debug_info))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error debugging SDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SDF debugging failed: {str(e)}")

@app.post("/test/model")
async def test_model(threshold: float = Query(MODEL_THRESHOLD, description="Classification threshold", ge=0.0, le=1.0)):
    """Test model with validation data"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="BBB prediction model not loaded")
        
        result = test_external_molecules(threshold)
        return JSONResponse(content=convert_numpy_types(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model testing failed: {str(e)}")

@app.get("/model/features")
async def get_model_features():
    """Get list of features used by the model"""
    try:
        return {
            "features": SELECTED_FEATURES,
            "feature_count": len(SELECTED_FEATURES),
            "descriptor_library": "Mordred"
        }
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving features: {str(e)}")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "File not found", "detail": str(exc)}
    )

@app.on_event("startup")
async def startup_event():
    """Startup event to check model availability"""
    logger.info("Starting BBB Prediction API...")
    logger.info(f"Environment: {'Development' if DEBUG else 'Production'}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Host: {HOST}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Default Threshold: {MODEL_THRESHOLD}")
    logger.info(f"Max File Size: {MAX_FILE_SIZE_MB}MB")
    logger.info(f"Max Batch Size: {MAX_BATCH_SIZE}")
    logger.info(f"Allowed Origins: {ALLOWED_ORIGINS}")
    
    if model is None:
        logger.warning("BBB prediction model not loaded!")
    else:
        logger.info("BBB prediction model loaded successfully")
    
    if not MORDRED_AVAILABLE:
        logger.warning("Mordred library not available!")
    else:
        logger.info("Mordred library available")
    
    logger.info("BBB Prediction API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("Shutting down BBB Prediction API...")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
>>>>>>> 65faad4285a7f91dd166127ff3c126d1bc2178b0
    )