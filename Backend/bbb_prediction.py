<<<<<<< HEAD
"""BBB permeability prediction using the trained LightGBM deployment bundle.

This module is intentionally standalone. Keep it in the same directory as
the uploaded workflow files, or keep those files in ``attached_assets/``:

    lgbm_bbb_model.joblib
    full_bbb_workflow.py
    preprocess_Descriptorcalc.py

The timestamped filenames uploaded to this Repl are also supported
automatically.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "attached_assets"

MODEL_FILENAMES = (
    "lgbm_bbb_model.joblib",
)

WORKFLOW_FILENAMES = (
    "full_bbb_workflow.py",
)

PREPROCESS_FILENAMES = (
    "preprocess_Descriptorcalc.py",
)

REQUIRED_BUNDLE_KEYS = {
    "pipeline",
    "retained_descriptors",
    "threshold",
    "descriptor_name_map",
    "X_train_imp",
    "y_train",
    "oof_proba_train",
}


def _candidate_paths(filenames: tuple[str, ...]) -> list[Path]:
    return [BASE_DIR / name for name in filenames] + [
        ASSET_DIR / name for name in filenames
    ]


def resolve_model_path(model_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit model path or find the uploaded model automatically."""
    configured = model_path or os.environ.get("BBB_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"BBB model file not found: {path}")
        return path

    for path in _candidate_paths(MODEL_FILENAMES):
        if path.is_file():
            return path

    searched = ", ".join(str(path) for path in _candidate_paths(MODEL_FILENAMES))
    raise FileNotFoundError(f"BBB model file not found. Searched: {searched}")


def _load_module(filename_candidates: tuple[str, ...], module_name: str):
    for path in _candidate_paths(filename_candidates):
        if not path.is_file():
            continue

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Python module: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    searched = ", ".join(str(path) for path in _candidate_paths(filename_candidates))
    raise FileNotFoundError(f"BBB workflow module not found. Searched: {searched}")


@lru_cache(maxsize=1)
def _deployment_functions() -> tuple[
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    """Load the original deployment workflow and preprocessing functions."""
    workflow = _load_module(WORKFLOW_FILENAMES, "full_bbb_workflow")
    preprocessing = _load_module(PREPROCESS_FILENAMES, "preprocess_Descriptorcalc")

    def mordred_fn(mol: Any) -> pd.DataFrame:
        from mordred import Calculator, descriptors

        calculator = Calculator(descriptors, ignore_3D=False)
        return pd.DataFrame([calculator(mol).asdict()])

    return (
        workflow.score_compound_for_deployment,
        preprocessing.embed_only,
        preprocessing.minimize_only,
        mordred_fn,
    )


@lru_cache(maxsize=1)
def load_bundle(model_path: str | None = None) -> dict[str, Any]:
    """Load and validate the trained model bundle once per process."""
    path = resolve_model_path(model_path)
    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError("The saved model must be a deployment bundle dictionary")

    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(
            "The model bundle is missing required fields: "
            f"{sorted(missing)}"
        )

    bundle["_model_path"] = str(path)
    return bundle


def mordred_descriptor(mol: Any) -> pd.DataFrame:
    """Calculate the full Mordred descriptor row used during training."""
    return _deployment_functions()[3](mol)


def curate_smiles(smiles: str) -> tuple[str | None, str]:
    """Validate and canonicalize a query SMILES before 3D generation.

    The deployment scorer accepts an optional curation callback.  Leaving that
    callback unset makes every otherwise valid molecule report
    ``skipped_no_curate_fn``, which hides whether the input actually passed
    curation.  Keep this step deliberately small and deterministic: parse the
    molecule, keep the largest organic fragment when a salt is supplied, and
    return RDKit's canonical SMILES.
    """
    from rdkit import Chem

    raw_smiles = smiles.strip()
    molecule = Chem.MolFromSmiles(raw_smiles)
    if molecule is None:
        return None, "invalid_smiles"

    fragments = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    organic_fragments = [
        fragment
        for fragment in fragments
        if any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
    ]
    if not organic_fragments:
        return None, "rejected_no_organic_fragment"

    curated = max(
        organic_fragments,
        key=lambda fragment: (fragment.GetNumHeavyAtoms(), fragment.GetNumAtoms()),
    )
    canonical_smiles = Chem.MolToSmiles(curated, canonical=True)
    if not canonical_smiles:
        return None, "curation_failed_empty_smiles"

    return canonical_smiles, "ok"


def score(
    smiles: str,
    bundle: dict[str, Any] | None = None,
    threshold: float | None = None,
    curate_fn: Callable[..., Any] | None = curate_smiles,
) -> dict[str, Any]:
    """Score one SMILES string using the exact saved deployment pipeline."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    active_bundle = bundle or load_bundle()
    active_threshold = (
        float(active_bundle["threshold"])
        if threshold is None
        else float(threshold)
    )
    if not 0 <= active_threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    score_compound, embed_only, minimize_only, descriptor_fn = _deployment_functions()
    return score_compound(
        smiles.strip(),
        active_bundle["pipeline"],
        active_bundle["retained_descriptors"],
        active_threshold,
        active_bundle["X_train_imp"],
        active_bundle["y_train"],
        active_bundle["oof_proba_train"],
        embed_fn=embed_only,
        minimize_fn=minimize_only,
        descriptor_fn=descriptor_fn,
        curate_fn=curate_fn,
        descriptor_name_map=active_bundle["descriptor_name_map"],
    )


def score_many(
    smiles_input: Any,
    bundle: dict[str, Any] | None = None,
    threshold: float | None = None,
    curate_fn: Callable[..., Any] | None = curate_smiles,
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    """Score a list, Series, DataFrame, or CSV path of SMILES strings."""
    active_bundle = bundle or load_bundle()
    active_threshold = (
        float(active_bundle["threshold"])
        if threshold is None
        else float(threshold)
    )
    if not 0 <= active_threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    score_compounds = _load_module(
        WORKFLOW_FILENAMES, "full_bbb_workflow"
    ).score_compounds_for_deployment
    _, embed_only, minimize_only, descriptor_fn = _deployment_functions()
    return score_compounds(
        smiles_input,
        active_bundle["pipeline"],
        active_bundle["retained_descriptors"],
        active_threshold,
        active_bundle["X_train_imp"],
        active_bundle["y_train"],
        active_bundle["oof_proba_train"],
        embed_fn=embed_only,
        minimize_fn=minimize_only,
        descriptor_fn=descriptor_fn,
        curate_fn=curate_fn,
        descriptor_name_map=active_bundle["descriptor_name_map"],
        smiles_col=smiles_col,
        verbose=False,
    )


def model_info(bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return metadata about the loaded model without running a prediction."""
    active_bundle = bundle or load_bundle()
    pipeline = active_bundle["pipeline"]
    return {
        "status": "loaded",
        "model_path": active_bundle.get("_model_path"),
        "model_type": pipeline.get("model_name", "LightGBM"),
        "imputer": pipeline.get("imp_name"),
        "feature_count": len(active_bundle["retained_descriptors"]),
        "retained_descriptors": active_bundle["retained_descriptors"],
        "threshold": float(active_bundle["threshold"]),
        "descriptor_library": "Mordred",
        "supports_3d_descriptors": True,
        "model_classes": ["BBB-", "BBB+"],
    }


def to_jsonable(value: Any) -> Any:
    """Convert NumPy, pandas, and NaN values into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def batch_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    successful = [row for row in rows if row.get("Status") == "Success"]
    positive = [row for row in successful if row.get("Prediction") == "BBB+"]
    return {
        "total_molecules": len(rows),
        "successful_predictions": len(successful),
        "failed_predictions": len(rows) - len(successful),
        "bbb_positive": len(positive),
        "bbb_negative": len(successful) - len(positive),
        "success_rate": len(successful) / len(rows) * 100 if rows else 0,
        "bbb_positive_rate": len(positive) / len(successful) * 100 if successful else 0,
        "threshold_used": threshold,
    }


def predict_batch(
    molecules: list[dict[str, Any]],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict a JSON batch and return rows plus summary statistics."""
    if not molecules:
        raise ValueError("molecules must be a non-empty list")

    bundle = load_bundle()
    active_threshold = (
        float(bundle["threshold"]) if threshold is None else float(threshold)
    )
    rows = []
    for index, molecule in enumerate(molecules):
        if not isinstance(molecule, dict):
            raise ValueError(f"molecules[{index}] must be an object")
        row = score(
            molecule.get("smiles", ""),
            bundle=bundle,
            threshold=active_threshold,
        )
        if molecule.get("name"):
            row["Name"] = str(molecule["name"])
        rows.append(row)

    return {
        "predictions": to_jsonable(rows),
        "summary": to_jsonable(batch_summary(rows, active_threshold)),
    }
=======
import sys
import json
import pandas as pd
import numpy as np
import joblib
import os
import io
from typing import List, Dict, Any, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcFractionCSP3
from rdkit.Chem import SDMolSupplier, SDWriter

# Import Mordred for descriptor calculation
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("Warning: Mordred not available. Please install with: pip install mordred")

# SELECTED MORDRED FEATURES (from your LightGBM model)
SELECTED_FEATURES = [
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

# UTILITY FUNCTIONS

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
            mol = Chem.rdMolStandardize.FragmentParent(mol)
            smiles = Chem.MolToSmiles(mol)
    except:
        pass
    
    return smiles

# FILE PROCESSING FUNCTIONS

def read_csv_file(file_content: str, smiles_column: str = None, name_column: str = None) -> List[Dict[str, str]]:
    """Read molecules from CSV content"""
    try:
        df = pd.read_csv(io.StringIO(file_content))
        
        # Auto-detect SMILES column if not specified
        if smiles_column is None:
            smiles_cols = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure', 'mol']
            smiles_column = None
            for col in smiles_cols:
                if col in df.columns:
                    smiles_column = col
                    break
            
            if smiles_column is None:
                smiles_column = df.columns[0]
        
        # Auto-detect name column if not specified
        if name_column is None:
            name_cols = ['name', 'Name', 'compound_name', 'title', 'id', 'ID', 'compound_id']
            name_column = None
            for col in name_cols:
                if col in df.columns:
                    name_column = col
                    break
        
        molecules = []
        for idx, row in df.iterrows():
            smiles = clean_smiles(row[smiles_column]) if smiles_column in row else ""
            name = str(row[name_column]) if name_column and name_column in row else f"Compound_{idx+1}"
            
            if smiles:
                molecules.append({
                    'smiles': smiles,
                    'name': name,
                    'row_index': idx
                })
        
        return molecules
        
    except Exception as e:
        raise ValueError(f"Error reading CSV: {str(e)}")

def read_sdf_file(file_content: str) -> List[Dict[str, Any]]:
    """Read molecules from SDF content"""
    try:
        molecules = []
        file_content = file_content.replace('\r\n', '\n').replace('\r', '\n')
        mol_blocks = file_content.split('\n$$$$\n')
        
        if len(mol_blocks) == 1 and '$$$$' in file_content:
            if '\n$$$$' in file_content:
                mol_blocks = file_content.split('\n$$$$')
            elif '$$$$\n' in file_content:
                mol_blocks = file_content.split('$$$$\n')
            else:
                mol_blocks = file_content.split('$$$$')
        
        for idx, mol_block in enumerate(mol_blocks):
            mol_block = mol_block.strip()
            if not mol_block:
                continue
            
            mol_block = mol_block.replace('$$$$', '').strip()
            if not mol_block:
                continue
                
            try:
                mol = Chem.MolFromMolBlock(mol_block)
                
                if mol is not None:
                    name = f"Molecule_{idx+1}"
                    
                    if mol.HasProp('_Name') and mol.GetProp('_Name').strip():
                        name = mol.GetProp('_Name').strip()
                    
                    mol_lines = mol_block.split('\n')
                    if len(mol_lines) > 0 and mol_lines[0].strip():
                        first_line = mol_lines[0].strip()
                        if first_line and not first_line.replace(' ', '').isdigit():
                            if not mol.HasProp('_Name') or not mol.GetProp('_Name').strip():
                                name = first_line
                    
                    smiles = Chem.MolToSmiles(mol)
                    
                    properties = {}
                    for prop_name in mol.GetPropNames():
                        try:
                            prop_value = mol.GetProp(prop_name)
                            properties[prop_name] = prop_value
                        except:
                            pass
                    
                    molecules.append({
                        'smiles': smiles,
                        'name': name,
                        'molBlock': mol_block,
                        'properties': properties,
                        'mol_index': idx
                    })
                    
                else:
                    molecules.append({
                        'smiles': None,
                        'name': f"Invalid_Molecule_{idx+1}",
                        'molBlock': mol_block,
                        'properties': {},
                        'error': "Failed to parse MOL block - invalid structure",
                        'mol_index': idx
                    })
                    
            except Exception as e:
                molecules.append({
                    'smiles': None,
                    'name': f"Error_Molecule_{idx+1}",
                    'molBlock': mol_block,
                    'properties': {},
                    'error': f"MOL block parsing error: {str(e)}",
                    'mol_index': idx
                })
        
        return molecules
        
    except Exception as e:
        raise ValueError(f"Error reading SDF: {str(e)}")

def read_mol_file(file_content: str, name: str = "Unknown") -> Dict[str, Any]:
    """Read single molecule from MOL file content"""
    try:
        mol = Chem.MolFromMolBlock(file_content)
        
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            
            properties = {}
            for prop_name in mol.GetPropNames():
                try:
                    properties[prop_name] = mol.GetProp(prop_name)
                except:
                    pass
            
            if mol.HasProp('_Name') and mol.GetProp('_Name').strip():
                name = mol.GetProp('_Name')
            
            return {
                'smiles': smiles,
                'name': name,
                'molBlock': file_content,
                'properties': properties
            }
        else:
            return {
                'smiles': None,
                'name': name,
                'molBlock': file_content,
                'error': "Failed to parse MOL structure"
            }
            
    except Exception as e:
        return {
            'smiles': None,
            'name': name,
            'molBlock': file_content,
            'error': f"MOL processing error: {str(e)}"
        }

def debug_sdf_content(file_content: str, max_chars: int = 1000) -> Dict[str, Any]:
    """Debug function to analyze SDF file content structure"""
    info = {
        'total_length': len(file_content),
        'total_lines': len(file_content.split('\n')),
        'delimiter_count': file_content.count('$$$$'),
        'first_chars': file_content[:max_chars],
        'last_chars': file_content[-max_chars:] if len(file_content) > max_chars else file_content,
        'line_endings': 'CRLF' if '\r\n' in file_content else 'LF' if '\n' in file_content else 'None',
        'empty_lines': file_content.count('\n\n'),
    }
    
    # Find positions of $$$$ delimiters
    delimiter_positions = []
    start = 0
    while True:
        pos = file_content.find('$$$$', start)
        if pos == -1:
            break
        delimiter_positions.append(pos)
        start = pos + 4
    
    info['delimiter_positions'] = delimiter_positions[:10]
    
    # Try to identify potential issues
    issues = []
    if info['delimiter_count'] == 0:
        issues.append("No $$$$ delimiters found - may not be a valid SDF file")
    
    if '\r\n' in file_content:
        issues.append("File contains Windows line endings (CRLF)")
    
    if not file_content.strip():
        issues.append("File appears to be empty")
    
    info['potential_issues'] = issues
    
    return info

# MORDRED DESCRIPTOR FUNCTIONS

def calculate_mordred_descriptors(smiles):
    """Calculate Mordred descriptors for selected features"""
    if not MORDRED_AVAILABLE:
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Initialize Mordred calculator
        calc = Calculator(descriptors, ignore_3D=True)
        
        # Calculate all descriptors
        desc_values = calc(mol)
        
        # Create descriptor dictionary with all calculated values
        desc_dict = {}
        for desc, value in zip(calc.descriptors, desc_values):
            desc_name = str(desc)
            try:
                # Handle different types of descriptor values
                if hasattr(value, 'real'):  # Complex numbers
                    desc_dict[desc_name] = float(value.real)
                elif pd.isna(value) or value is None:
                    desc_dict[desc_name] = np.nan
                else:
                    desc_dict[desc_name] = float(value)
            except (ValueError, TypeError):
                desc_dict[desc_name] = np.nan
        
        # Convert to DataFrame to handle processing
        desc_df = pd.DataFrame([desc_dict])
        
        # Handle Lipinski and GhoseFilter - convert to boolean (int)
        for col in ["Lipinski", "GhoseFilter"]:
            if col in desc_df.columns:
                desc_df[col] = desc_df[col].astype(int)
        
        # Convert to numeric and fill NaN with median (or 0 for single molecule)
        desc_df = desc_df.apply(pd.to_numeric, errors='coerce')
        desc_df = desc_df.fillna(0)  # Use 0 for single molecule prediction
        
        # Extract only the selected features
        selected_desc = {}
        for feature in SELECTED_FEATURES:
            if feature in desc_df.columns:
                selected_desc[feature] = desc_df[feature].iloc[0]
            else:
                selected_desc[feature] = 0.0  # Default value for missing features
        
        return selected_desc
        
    except Exception as e:
        print(f"Error calculating Mordred descriptors: {str(e)}")
        return None

def validate_smiles(smiles):
    """Validate SMILES string"""
    try:
        if not smiles or pd.isna(smiles):
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False

# MODEL LOADING

def load_bbb_model(model_path='lgbm_model.pkl'):
    """Load trained LightGBM BBB classification model"""
    try:
        model = joblib.load(model_path)
        return model, SELECTED_FEATURES
    except FileNotFoundError:
        print(f"Model file {model_path} not found")
        return None, None

# Load the model globally
model, feature_names = load_bbb_model()

# PREDICTION FUNCTIONS

def predict_single_molecule(smiles, name="Unknown", threshold=0.5228):
    """Predict BBB permeability for a single SMILES using Mordred descriptors"""
    
    if not MORDRED_AVAILABLE:
        return {
            'smiles': smiles,
            'name': name,
            'status': 'Error',
            'prediction': None,
            'probability_bbb_negative': None,
            'probability_bbb_positive': None,
            'confidence': None,
            'interpretation': 'Mordred library not installed',
            'error': 'Mordred library not available'
        }
    
    # Clean and validate SMILES
    smiles = clean_smiles(smiles)
    if not validate_smiles(smiles):
        return {
            'smiles': smiles,
            'name': name,
            'status': 'Error',
            'prediction': None,
            'probability_bbb_negative': None,
            'probability_bbb_positive': None,
            'confidence': None,
            'interpretation': 'Invalid SMILES structure',
            'error': 'Invalid SMILES'
        }
    
    # Calculate Mordred descriptors
    desc = calculate_mordred_descriptors(smiles)
    if desc is None:
        return {
            'smiles': smiles,
            'name': name,
            'status': 'Error',
            'prediction': None,
            'probability_bbb_negative': None,
            'probability_bbb_positive': None,
            'confidence': None,
            'interpretation': 'Failed to calculate Mordred descriptors',
            'error': 'Descriptor calculation failed'
        }
    
    # Prepare features for prediction
    try:
        # Create feature vector in the correct order
        X_input = []
        for feature in SELECTED_FEATURES:
            X_input.append(desc.get(feature, 0.0))
        
        X_input = np.array(X_input).reshape(1, -1)
        
        # Make prediction
        probabilities = model.predict_proba(X_input)[0]
        prob_bbb_negative = float(probabilities[0])  # Class 0 (BBB-)
        prob_bbb_positive = float(probabilities[1])  # Class 1 (BBB+)
        
        # Apply custom threshold
        prediction_binary = 1 if prob_bbb_positive >= threshold else 0
        prediction = 'BBB+' if prediction_binary == 1 else 'BBB-'
        
        # Calculate confidence (max probability)
        confidence = float(max(probabilities))
        
        # Generate interpretation
        if confidence >= 0.8:
            confidence_level = "High"
        elif confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        interpretation = f"{prediction} ({confidence_level} confidence, threshold={threshold})"
        
        result = {
            'smiles': smiles,
            'name': name,
            'status': 'Success',
            'prediction': prediction,
            'probability_bbb_negative': prob_bbb_negative,
            'probability_bbb_positive': prob_bbb_positive,
            'confidence': confidence,
            'threshold_used': threshold,
            'interpretation': interpretation,
            'mordred_descriptors': desc
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        return {
            'smiles': smiles,
            'name': name,
            'status': 'Error',
            'prediction': None,
            'probability_bbb_negative': None,
            'probability_bbb_positive': None,
            'confidence': None,
            'interpretation': f'Prediction failed: {str(e)}',
            'error': str(e)
        }

def predict_multiple_molecules(molecules, threshold=0.5228):
    """Predict BBB permeability for multiple molecules"""
    results = []
    
    for molecule in molecules:
        smiles = molecule.get('smiles', '')
        name = molecule.get('name', 'Unknown')
        result = predict_single_molecule(smiles, name, threshold)
        
        # Add original molecule info
        result['original_data'] = molecule
        results.append(result)
    
    return results

def predict_from_file(file_content: str, file_type: str, threshold: float = 0.5228, **kwargs) -> Dict[str, Any]:
    """Predict BBB permeability from file content"""
    try:
        molecules = []
        
        if file_type.lower() == 'csv':
            molecules = read_csv_file(
                file_content, 
                smiles_column=kwargs.get('smiles_column'),
                name_column=kwargs.get('name_column')
            )
            
        elif file_type.lower() == 'sdf':
            molecules = read_sdf_file(file_content)
            
        elif file_type.lower() == 'mol':
            molecule = read_mol_file(file_content, kwargs.get('name', 'Unknown'))
            molecules = [molecule]
            
        else:
            return {
                'success': False,
                'error': f"Unsupported file type: {file_type}. Supported types: csv, sdf, mol"
            }
        
        if not molecules:
            return {
                'success': False,
                'error': f"No valid molecules found in {file_type.upper()} file"
            }
        
        # Make predictions
        predictions = predict_multiple_molecules(molecules, threshold)
        
        # Calculate summary statistics
        successful = [p for p in predictions if p['status'] == 'Success']
        bbb_positive = [p for p in successful if p['prediction'] == 'BBB+']
        bbb_negative = [p for p in successful if p['prediction'] == 'BBB-']
        high_conf = [p for p in successful if p['confidence'] >= 0.8]
        medium_conf = [p for p in successful if 0.6 <= p['confidence'] < 0.8]
        low_conf = [p for p in successful if p['confidence'] < 0.6]
        
        summary = {
            'file_type': file_type.upper(),
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
            'threshold_used': threshold
        }
        
        summary = convert_numpy_types(summary)
        
        return {
            'success': True,
            'data': {
                'predictions': predictions,
                'summary': summary,
                'file_info': {
                    'type': file_type.upper(),
                    'molecules_processed': len(molecules)
                }
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error processing {file_type.upper()} file: {str(e)}"
        }

def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {
            'status': 'Model not loaded',
            'error': 'LightGBM BBB classification model not found'
        }
    
    model_info = {
        'status': 'Model loaded successfully',
        'model_type': 'LightGBM',
        'feature_count': len(SELECTED_FEATURES),
        'features': SELECTED_FEATURES,
        'descriptor_library': 'Mordred',
        'mordred_available': MORDRED_AVAILABLE,
        'model_classes': ['BBB-', 'BBB+'],
        'default_threshold': 0.5228,
        'supported_file_types': ['CSV', 'SDF', 'MOL'],
        'supported_actions': [
            'get_model_info',
            'predict_single',
            'predict_batch',
            'predict_from_file',
            'debug_sdf',
            'test_external_molecules'
        ]
    }
    
    return convert_numpy_types(model_info)

def test_external_molecules(threshold=0.5228):
    """Test the model with a simple validation - removed external test set"""
    return {
        'status': 'External test set validation disabled',
        'message': 'Model is ready for prediction without external validation',
        'threshold_used': threshold,
        'model_status': 'Loaded and operational'
    }

# MAIN FUNCTION

def main():
    """Main function to handle requests"""
    
    if model is None:
        print(json.dumps({
            "success": False,
            "error": "LightGBM BBB classification model not loaded. Please ensure lgbm_model.pkl exists."
        }))
        sys.exit(1)
    
    if not MORDRED_AVAILABLE:
        print(json.dumps({
            "success": False,
            "error": "Mordred library not available. Please install with: pip install mordred"
        }))
        sys.exit(1)
    
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        action = input_data.get('action')
        threshold = input_data.get('threshold', 0.5228)
        
        if action == 'get_model_info':
            result = {
                "success": True,
                "data": get_model_info()
            }
            
        elif action == 'predict_single':
            smiles = input_data.get('smiles')
            name = input_data.get('name', 'Unknown')
            
            prediction = predict_single_molecule(smiles, name, threshold)
            result = {
                "success": True,
                "data": prediction
            }
            
        elif action == 'predict_batch':
            molecules = input_data.get('molecules', [])
            
            predictions = predict_multiple_molecules(molecules, threshold)
            
            # Calculate summary statistics
            successful = [p for p in predictions if p['status'] == 'Success']
            bbb_positive = [p for p in successful if p['prediction'] == 'BBB+']
            bbb_negative = [p for p in successful if p['prediction'] == 'BBB-']
            high_conf = [p for p in successful if p['confidence'] >= 0.8]
            
            summary = {
                'total_molecules': len(predictions),
                'successful_predictions': len(successful),
                'failed_predictions': len(predictions) - len(successful),
                'bbb_positive': len(bbb_positive),
                'bbb_negative': len(bbb_negative),
                'high_confidence_predictions': len(high_conf),
                'success_rate': len(successful) / len(predictions) * 100 if predictions else 0,
                'bbb_positive_rate': len(bbb_positive) / len(successful) * 100 if successful else 0,
                'threshold_used': threshold
            }
            
            summary = convert_numpy_types(summary)
            
            result = {
                "success": True,
                "data": {
                    "predictions": predictions,
                    "summary": summary
                }
            }
            
        elif action == 'predict_from_file':
            file_content = input_data.get('file_content')
            file_type = input_data.get('file_type')
            
            # Optional parameters for CSV parsing
            kwargs = {}
            if 'smiles_column' in input_data:
                kwargs['smiles_column'] = input_data['smiles_column']
            if 'name_column' in input_data:
                kwargs['name_column'] = input_data['name_column']
            if 'name' in input_data:
                kwargs['name'] = input_data['name']
                
            result = predict_from_file(file_content, file_type, threshold, **kwargs)
            
        elif action == 'debug_sdf':
            file_content = input_data.get('file_content')
            debug_info = debug_sdf_content(file_content)
            
            # Try parsing with current method
            try:
                molecules = read_sdf_file(file_content)
                debug_info['parsing_result'] = {
                    'success': True,
                    'molecules_found': len(molecules),
                    'valid_molecules': len([m for m in molecules if m.get('smiles')])
                }
            except Exception as e:
                debug_info['parsing_result'] = {
                    'success': False,
                    'error': str(e)
                }
            
            result = {
                'success': True,
                'data': debug_info
            }
            
        elif action == 'test_external_molecules':
            test_results = test_external_molecules(threshold)
            result = {
                "success": True,
                "data": test_results
            }
            
        else:
            result = {
                "success": False,
                "error": f"Unknown action: {action}. Supported actions: {', '.join(get_model_info()['supported_actions'])}"
            }
        
        # Final conversion of entire result to ensure JSON serialization
        result = convert_numpy_types(result)
        print(json.dumps(result))
        
    except json.JSONDecodeError:
        print(json.dumps({
            "success": False,
            "error": "Invalid JSON input"
        }))
        sys.exit(1)
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
>>>>>>> 65faad4285a7f91dd166127ff3c126d1bc2178b0
