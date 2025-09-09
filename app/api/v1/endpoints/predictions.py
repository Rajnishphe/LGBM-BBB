# =============================================================================
# app/api/v1/endpoints/predictions.py
# =============================================================================
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import base64
import logging

from models.schemas import (
    MoleculeInput,
    BatchMoleculeInput,
    FileUploadInput,
    PredictionResult,
    BatchPredictionResponse,
    ModelInfoResponse
)
from services.prediction_service import prediction_service
from utils.file_handlers import read_csv_molecules, read_sdf_molecules, read_mol_molecule
from core.exceptions import BBBPredictionException

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    try:
        model_info = prediction_service.get_model_info()
        return ModelInfoResponse(success=True, data=model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return ModelInfoResponse(success=False, error=str(e))

@router.post("/predict/single", response_model=BatchPredictionResponse)
async def predict_single_molecule(molecule: MoleculeInput, threshold: float = 0.5228):
    """Predict BBB permeability for a single molecule"""
    try:
        result = prediction_service.predict_single(
            smiles=molecule.smiles,
            name=molecule.name,
            threshold=threshold
        )
        
        return BatchPredictionResponse(
            success=True,
            data={"prediction": result.dict()}
        )
        
    except BBBPredictionException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_molecules(batch_input: BatchMoleculeInput):
    """Predict BBB permeability for multiple molecules"""
    try:
        molecules = [mol.dict() for mol in batch_input.molecules]
        results = prediction_service.predict_batch(molecules, batch_input.threshold)
        
        # Calculate summary statistics
        successful = [r for r in results if r.status == 'Success']
        bbb_positive = [r for r in successful if r.prediction == 'BBB+']
        bbb_negative = [r for r in successful if r.prediction == 'BBB-']
        high_conf = [r for r in successful if r.confidence and r.confidence >= 0.8]
        medium_conf = [r for r in successful if r.confidence and 0.6 <= r.confidence < 0.8]
        low_conf = [r for r in successful if r.confidence and r.confidence < 0.6]
        
        summary = {
            'total_molecules': len(results),
            'successful_predictions': len(successful),
            'failed_predictions': len(results) - len(successful),
            'bbb_positive': len(bbb_positive),
            'bbb_negative': len(bbb_negative),
            'high_confidence': len(high_conf),
            'medium_confidence': len(medium_conf),
            'low_confidence': len(low_conf),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'bbb_positive_rate': len(bbb_positive) / len(successful) * 100 if successful else 0,
            'threshold_used': batch_input.threshold
        }
        
        return BatchPredictionResponse(
            success=True,
            data={
                "predictions": [result.dict() for result in results],
                "summary": summary
            }
        )
        
    except BBBPredictionException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/file", response_model=BatchPredictionResponse)
async def predict_from_file(file_input: FileUploadInput):
    """Predict BBB permeability from uploaded file"""
    try:
        # Decode base64 content
        try:
            file_content = base64.b64decode(file_input.file_content).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file content: {str(e)}")
        
        # Process file based on type
        molecules = []
        
        if file_input.file_type == "csv":
            molecules = read_csv_molecules(
                file_content,
                smiles_column=file_input.smiles_column,
                name_column=file_input.name_column
            )
        elif file_input.file_type == "sdf":
            molecules = read_sdf_molecules(file_content)
        elif file_input.file_type == "mol":
            molecule = read_mol_molecule(file_content, file_input.molecule_name)
            molecules = [molecule]
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_input.file_type}"
            )
        
        if not molecules:
            raise HTTPException(
                status_code=400,
                detail=f"No valid molecules found in {file_input.file_type.upper()} file"
            )
        
        # Make predictions
        results = prediction_service.predict_batch(molecules, file_input.threshold)
        
        # Calculate summary statistics
        successful = [r for r in results if r.status == 'Success']
        bbb_positive = [r for r in successful if r.prediction == 'BBB+']
        bbb_negative = [r for r in successful if r.prediction == 'BBB-']
        high_conf = [r for r in successful if r.confidence and r.confidence >= 0.8]
        medium_conf = [r for r in successful if r.confidence and 0.6 <= r.confidence < 0.8]
        low_conf = [r for r in successful if r.confidence and r.confidence < 0.6]
        
        summary = {
            'file_type': file_input.file_type.upper(),
            'total_molecules': len(results),
            'successful_predictions': len(successful),
            'failed_predictions': len(results) - len(successful),
            'bbb_positive': len(bbb_positive),
            'bbb_negative': len(bbb_negative),
            'high_confidence': len(high_conf),
            'medium_confidence': len(medium_conf),
            'low_confidence': len(low_conf),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'bbb_positive_rate': len(bbb_positive) / len(successful) * 100 if successful else 0,
            'threshold_used': file_input.threshold
        }
        
        return BatchPredictionResponse(
            success=True,
            data={
                "predictions": [result.dict() for result in results],
                "summary": summary,
                "file_info": {
                    "type": file_input.file_type.upper(),
                    "molecules_processed": len(molecules)
                }
            }
        )
        
    except BBBPredictionException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))