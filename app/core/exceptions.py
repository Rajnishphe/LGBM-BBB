from typing import Optional

class BBBPredictionException(Exception):
    """Base exception for BBB prediction errors"""
    def __init__(
        self, 
        message: str, 
        status_code: int = 400, 
        detail: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)

class ModelNotLoadedException(BBBPredictionException):
    """Raised when the ML model is not loaded"""
    def __init__(self, detail: Optional[str] = None):
        super().__init__(
            message="Machine learning model not loaded",
            status_code=500,
            detail=detail
        )

class InvalidMoleculeException(BBBPredictionException):
    """Raised when molecule structure is invalid"""
    def __init__(self, smiles: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Invalid molecule structure: {smiles}",
            status_code=400,
            detail=detail
        )

class FileProcessingException(BBBPredictionException):
    """Raised when file processing fails"""
    def __init__(self, file_type: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Failed to process {file_type} file",
            status_code=400,
            detail=detail
        )

class DescriptorCalculationException(BBBPredictionException):
    """Raised when descriptor calculation fails"""
    def __init__(self, detail: Optional[str] = None):
        super().__init__(
            message="Failed to calculate molecular descriptors",
            status_code=500,
            detail=detail
        )
