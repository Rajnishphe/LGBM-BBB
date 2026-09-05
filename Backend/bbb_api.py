"""HTTP entry point for the bundled BBB permeability backend."""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bbb_prediction import (
    batch_summary,
    load_bundle,
    model_info,
    predict_batch,
    score,
    to_jsonable,
)


PORT = int(os.environ.get("PORT", "8000"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1000"))
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bbb-api")

app = FastAPI(
    title="BBB Permeability Prediction API",
    description=(
        "Blood-brain barrier permeability prediction using the trained "
        "LightGBM bundle."
    ),
    version="2.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def accept_api_prefix(request: Request, call_next):
    """Accept the Replit artifact's /api proxy prefix without losing routes."""
    path = request.scope.get("path", "")
    if path == "/api" or path.startswith("/api/"):
        request.scope["path"] = path[4:] or "/"
        request.scope["root_path"] = "/api"
    return await call_next(request)


class SinglePredictionRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    name: str = "Unknown"
    threshold: float | None = Field(default=None, ge=0, le=1)


class SmilesValidationRequest(BaseModel):
    smiles: str = Field(..., min_length=1)


class MoleculeRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    name: str | None = None


class BatchPredictionRequest(BaseModel):
    molecules: list[MoleculeRequest] = Field(..., min_length=1, max_length=1000)
    threshold: float | None = Field(default=None, ge=0, le=1)


def current_bundle() -> dict[str, Any]:
    try:
        return load_bundle()
    except Exception as error:
        logger.exception("Unable to load BBB model")
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "BBB Permeability Prediction API",
        "version": "2.1.0",
        "status": "running",
    }


@app.get("/health")
@app.get("/healthz")
def health() -> dict[str, Any]:
    try:
        info = model_info(load_bundle())
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_loaded": True,
            "model_type": info["model_type"],
            "threshold": info["threshold"],
        }
    except Exception as error:
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_loaded": False,
            "error": str(error),
        }


@app.get("/model/info")
def get_model_info() -> dict[str, Any]:
    return to_jsonable(model_info(current_bundle()))


@app.post("/validate/smiles")
def validate_smiles(request: SmilesValidationRequest) -> dict[str, Any]:
    """Validate a SMILES string without loading the prediction model."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(request.smiles.strip())
    return {
        "valid": molecule is not None,
        "smiles": request.smiles,
        "canonical_smiles": (
            Chem.MolToSmiles(molecule) if molecule is not None else None
        ),
    }


@app.post("/predict/single")
def predict_single(request: SinglePredictionRequest) -> dict[str, Any]:
    bundle = current_bundle()
    try:
        result = score(
            request.smiles,
            bundle=bundle,
            threshold=request.threshold,
        )
        result["Name"] = request.name
        return to_jsonable(result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Single prediction failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchPredictionRequest) -> dict[str, Any]:
    if len(request.molecules) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size cannot exceed {MAX_BATCH_SIZE} molecules",
        )

    current_bundle()
    try:
        result = predict_batch(
            [molecule.model_dump() for molecule in request.molecules],
            threshold=request.threshold,
        )
        return to_jsonable(result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


def parse_uploaded_file(
    content: bytes,
    filename: str,
    smiles_column: str | None,
    name_column: str | None,
) -> tuple[list[dict[str, str]], str]:
    suffix = os.path.splitext(filename.lower())[1]

    if suffix == ".csv":
        try:
            dataframe = pd.read_csv(io.BytesIO(content))
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {error}") from error

        if dataframe.empty:
            raise HTTPException(status_code=400, detail="CSV contains no rows")

        selected_smiles_column = smiles_column or next(
            (
                column
                for column in ["SMILES", "smiles", "Smiles", "canonical_smiles", "structure"]
                if column in dataframe.columns
            ),
            str(dataframe.columns[0]),
        )
        if selected_smiles_column not in dataframe.columns:
            raise HTTPException(
                status_code=400,
                detail=f"SMILES column not found: {selected_smiles_column}",
            )

        if name_column and name_column not in dataframe.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Name column not found: {name_column}",
            )

        molecules = []
        for index, row in dataframe.iterrows():
            smiles = str(row[selected_smiles_column]).strip()
            if not smiles or smiles.lower() == "nan":
                continue
            molecules.append(
                {
                    "smiles": smiles,
                    "name": (
                        str(row[name_column])
                        if name_column
                        else f"Compound_{index + 1}"
                    ),
                }
            )
        return molecules, "csv"

    from rdkit import Chem

    if suffix == ".mol":
        mol = Chem.MolFromMolBlock(content.decode("utf-8", errors="replace"))
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid MOL file")
        return [{"smiles": Chem.MolToSmiles(mol), "name": filename}], "mol"

    if suffix == ".sdf":
        text = content.decode("utf-8", errors="replace")
        molecules = []
        for index, block in enumerate(text.split("$$$$")):
            block = block.strip()
            if not block:
                continue
            mol = Chem.MolFromMolBlock(block)
            if mol is not None:
                molecules.append(
                    {
                        "smiles": Chem.MolToSmiles(mol),
                        "name": (
                            mol.GetProp("_Name")
                            if mol.HasProp("_Name") and mol.GetProp("_Name").strip()
                            else f"Molecule_{index + 1}"
                        ),
                    }
                )
        if not molecules:
            raise HTTPException(status_code=400, detail="No valid molecules found in SDF")
        return molecules, "sdf"

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Use .csv, .sdf, or .mol",
    )


@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0, le=1),
    smiles_column: str | None = Query(default=None),
    name_column: str | None = Query(default=None),
) -> dict[str, Any]:
    filename = file.filename or "upload"
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB",
        )

    molecules, file_type = parse_uploaded_file(
        content, filename, smiles_column, name_column
    )
    if not molecules:
        raise HTTPException(status_code=400, detail="No molecules found in uploaded file")
    if len(molecules) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File contains more than {MAX_BATCH_SIZE} molecules",
        )

    bundle = current_bundle()
    try:
        rows = []
        for molecule in molecules:
            row = score(
                molecule["smiles"],
                bundle=bundle,
                threshold=threshold,
            )
            row["Name"] = molecule["name"]
            rows.append(row)
        active_threshold = (
            float(bundle["threshold"]) if threshold is None else float(threshold)
        )
        return to_jsonable(
            {
                "success": True,
                "file_type": file_type.upper(),
                "predictions": rows,
                "summary": batch_summary(rows, active_threshold),
            }
        )
    except Exception as error:
        logger.exception("File prediction failed")
        raise HTTPException(status_code=500, detail=str(error)) from error