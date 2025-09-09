# app/utils/molecular_utils.py
# =============================================================================
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolStandardize
import numpy as np

# Import Mordred for descriptor calculation
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False

def clean_smiles(smiles: str) -> str:
    """Clean and standardize SMILES string"""
    if not smiles or pd.isna(smiles):
        return ""
    
    # Remove common artifacts
    smiles = str(smiles).strip()
    smiles = smiles.replace('\n', '').replace('\r', '')
    
    # Remove salt components (keep largest fragment)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Remove salts and keep largest fragment
            mol = rdMolStandardize.FragmentParent(mol)
            smiles = Chem.MolToSmiles(mol)
    except:
        pass
    
    return smiles

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string"""
    try:
        if not smiles or pd.isna(smiles):
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False

def calculate_mordred_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """Calculate Mordred descriptors for selected features"""
    if not MORDRED_AVAILABLE:
        return None
    
    from core.config import settings
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Initialize Mordred calculator
        calc = Calculator(descriptors, ignore_3D=True)
        
        # Calculate all descriptors
        desc_values = calc(mol)
        
        # Create descriptor dictionary
        desc_dict = {}
        for desc, value in zip(calc.descriptors, desc_values):
            desc_name = str(desc)
            try:
                if hasattr(value, 'real'):  # Complex numbers
                    desc_dict[desc_name] = float(value.real)
                elif pd.isna(value) or value is None:
                    desc_dict[desc_name] = np.nan
                else:
                    desc_dict[desc_name] = float(value)
            except (ValueError, TypeError):
                desc_dict[desc_name] = np.nan
        
        # Convert to DataFrame for processing
        desc_df = pd.DataFrame([desc_dict])
        
        # Handle boolean descriptors
        for col in ["Lipinski", "GhoseFilter"]:
            if col in desc_df.columns:
                desc_df[col] = desc_df[col].astype(int)
        
        # Convert to numeric and fill NaN
        desc_df = desc_df.apply(pd.to_numeric, errors='coerce')
        desc_df = desc_df.fillna(0)
        
        # Extract selected features
        selected_desc = {}
        for feature in settings.SELECTED_FEATURES:
            if feature in desc_df.columns:
                selected_desc[feature] = desc_df[feature].iloc[0]
            else:
                selected_desc[feature] = 0.0
        
        return selected_desc
        
    except Exception as e:
        print(f"Error calculating Mordred descriptors: {str(e)}")
        return None

