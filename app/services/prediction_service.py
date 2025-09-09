import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from core.config import settings
from core.exceptions import (
    ModelNotLoadedException, 
    InvalidMoleculeException, 
    DescriptorCalculationException
)
from utils.molecular_utils import (
    clean_smiles, 
    validate_smiles, 
    calculate_mordred_descriptors
)
from models.schemas import PredictionResult

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.feature_names = settings.SELECTED_FEATURES
        self._load_model()
    
    def _load_model(self):
        """Load the trained LightGBM model"""
        try:
            model_path = Path(settings.MODEL_PATH)
            if not model_path.exists():
                raise ModelNotLoadedException(f"Model file not found: {model_path}")
            
            self.model = joblib.load(str(model_path))
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelNotLoadedException(f"Failed to load model: {str(e)}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_model_loaded():
            raise ModelNotLoadedException()
        
        return {
            'status': 'Model loaded successfully',
            'model_type': 'LightGBM',
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'descriptor_library': 'Mordred',
            'model_classes': ['BBB-', 'BBB+'],
            'default_threshold': settings.DEFAULT_THRESHOLD,
            'supported_file_types': settings.SUPPORTED_FILE_TYPES,
            'max_molecules_per_request': settings.MAX_MOLECULES_PER_REQUEST
        }
    
    def predict_single(
        self, 
        smiles: str, 
        name: str = "Unknown", 
        threshold: float = None
    ) -> PredictionResult:
        """Predict BBB permeability for a single molecule"""
        
        if not self.is_model_loaded():
            raise ModelNotLoadedException()
        
        if threshold is None:
            threshold = settings.DEFAULT_THRESHOLD
        
        # Clean and validate SMILES
        smiles = clean_smiles(smiles)
        if not validate_smiles(smiles):
            return PredictionResult(
                smiles=smiles,
                name=name,
                status='Error',
                error='Invalid SMILES structure',
                interpretation='Invalid SMILES structure'
            )
        
        try:
            # Calculate Mordred descriptors
            descriptors = calculate_mordred_descriptors(smiles)
            if descriptors is None:
                return PredictionResult(
                    smiles=smiles,
                    name=name,
                    status='Error',
                    error='Failed to calculate molecular descriptors',
                    interpretation='Failed to calculate Mordred descriptors'
                )
            
            # Prepare feature vector
            X_input = []
            for feature in self.feature_names:
                X_input.append(descriptors.get(feature, 0.0))
            
            X_input = np.array(X_input).reshape(1, -1)
            
            # Make prediction
            probabilities = self.model.predict_proba(X_input)[0]
            prob_bbb_negative = float(probabilities[0])  # Class 0 (BBB-)
            prob_bbb_positive = float(probabilities[1])  # Class 1 (BBB+)
            
            # Apply threshold
            prediction_binary = 1 if prob_bbb_positive >= threshold else 0
            prediction = 'BBB+' if prediction_binary == 1 else 'BBB-'
            
            # Calculate confidence
            confidence = float(max(probabilities))
            
            # Generate interpretation
            if confidence >= 0.8:
                confidence_level = "High"
            elif confidence >= 0.6:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            interpretation = f"{prediction} ({confidence_level} confidence, threshold={threshold})"
            
            return PredictionResult(
                smiles=smiles,
                name=name,
                status='Success',
                prediction=prediction,
                probability_bbb_negative=prob_bbb_negative,
                probability_bbb_positive=prob_bbb_positive,
                confidence=confidence,
                threshold_used=threshold,
                interpretation=interpretation,
                mordred_descriptors=descriptors
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for {smiles}: {str(e)}")
            return PredictionResult(
                smiles=smiles,
                name=name,
                status='Error',
                error=str(e),
                interpretation=f'Prediction failed: {str(e)}'
            )
    
    def predict_batch(
        self, 
        molecules: List[Dict[str, str]], 
        threshold: float = None
    ) -> List[PredictionResult]:
        """Predict BBB permeability for multiple molecules"""
        
        if threshold is None:
            threshold = settings.DEFAULT_THRESHOLD
        
        results = []
        for molecule in molecules:
            smiles = molecule.get('smiles', '')
            name = molecule.get('name', 'Unknown')
            result = self.predict_single(smiles, name, threshold)
            results.append(result)
        
        return results

# Global service instance
prediction_service = PredictionService()