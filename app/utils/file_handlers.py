# app/utils/file_handlers.py
# =============================================================================
import pandas as pd
import io
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter

from core.exceptions import FileProcessingException
from utils.molecular_utils import clean_smiles

def read_csv_molecules(
    file_content: str, 
    smiles_column: str = None, 
    name_column: str = None
) -> List[Dict[str, str]]:
    """Read molecules from CSV content"""
    try:
        df = pd.read_csv(io.StringIO(file_content))
        
        # Auto-detect SMILES column
        if smiles_column is None:
            smiles_cols = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure', 'mol']
            for col in smiles_cols:
                if col in df.columns:
                    smiles_column = col
                    break
            if smiles_column is None:
                smiles_column = df.columns[0]
        
        # Auto-detect name column
        if name_column is None:
            name_cols = ['name', 'Name', 'compound_name', 'title', 'id', 'ID', 'compound_id']
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
                    'name': name
                })
        
        return molecules
        
    except Exception as e:
        raise FileProcessingException("CSV", str(e))

def read_sdf_molecules(file_content: str) -> List[Dict[str, Any]]:
    """Read molecules from SDF content"""
    try:
        molecules = []
        file_content = file_content.replace('\r\n', '\n').replace('\r', '\n')
        mol_blocks = file_content.split('\n$$$$\n')
        
        # Handle different delimiter patterns
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
                    
                    smiles = Chem.MolToSmiles(mol)
                    
                    molecules.append({
                        'smiles': smiles,
                        'name': name
                    })
                    
            except Exception as e:
                molecules.append({
                    'smiles': None,
                    'name': f"Error_Molecule_{idx+1}",
                    'error': f"MOL block parsing error: {str(e)}"
                })
        
        return molecules
        
    except Exception as e:
        raise FileProcessingException("SDF", str(e))

def read_mol_molecule(file_content: str, name: str = "Unknown") -> Dict[str, Any]:
    """Read single molecule from MOL file content"""
    try:
        mol = Chem.MolFromMolBlock(file_content)
        
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            
            if mol.HasProp('_Name') and mol.GetProp('_Name').strip():
                name = mol.GetProp('_Name')
            
            return {
                'smiles': smiles,
                'name': name
            }
        else:
            return {
                'smiles': None,
                'name': name,
                'error': "Failed to parse MOL structure"
            }
            
    except Exception as e:
        raise FileProcessingException("MOL", str(e))