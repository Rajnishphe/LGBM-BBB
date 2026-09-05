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