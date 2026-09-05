"""
predict_bbb.py
---------------
Thin inference-only wrapper around the saved LGBM-BBB model bundle
(lgbm_bbb_model.joblib). This is what your Flask endpoints / adhoc
scoring scripts should import, instead of re-running training code.

Exposes:
    load_bundle(path) -> dict
    score(smiles, bundle) -> dict            (single compound)
    score_many(smiles_input, bundle) -> pd.DataFrame   (batch)

Requires full_bbb_workflow.py and preprocess_Descriptorcalc.py to be
importable (same folder, or on sys.path).
"""

import joblib
import pandas as pd

from full_bbb_workflow import score_compound_for_deployment, score_compounds_for_deployment
from preprocess_Descriptorcalc import embed_only, minimize_only

# Optional: wire these in if/when you have curation helpers available.
# Left as None here since canonicalize_smiles / final_curated_df aren't
# defined in preprocess_Descriptorcalc.py or full_bbb_workflow.py yet.
try:
    from preprocess_Descriptorcalc import canonicalize_smiles, final_curated_df

    def _curation_fn(smiles):
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return None, "invalid_smiles"
        single_row_df = pd.DataFrame({"canonicalisedSMILES": [canonical]})
        curated_df = final_curated_df(
            single_row_df, smiles_col="canonicalisedSMILES", exclude_simple_inorganic_carbon=True,
        )
        if len(curated_df) == 0:
            return None, "excluded_by_curation"
        return curated_df["canonicalisedSMILES"].iloc[0], "ok"

except ImportError:
    _curation_fn = None  # curation will be skipped and flagged in output


def _mordred_fn(mol):
    """One RDKit Mol in, one-row DataFrame of raw Mordred descriptors out."""
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=False)
    result = calc(mol)
    return pd.DataFrame([result.asdict()])


def load_bundle(path: str = "lgbm_bbb_model.joblib") -> dict:
    """Loads the joblib bundle saved by the training notebook (cell 7).

    Expected keys: pipeline, retained_descriptors, threshold,
    descriptor_name_map, X_train_imp, y_train, oof_proba_train.
    """
    return joblib.load(path)


def score(smiles: str, bundle: dict, weights: tuple = (0.5, 0.0, 0.5)) -> dict:
    """Scores a single SMILES string. Returns the same dict shape as
    score_compound_for_deployment(): SMILES, Curation_Status,
    3D_Generation_Status, Prediction, Confidence,
    BBB_plus_Probability_Percent, Status.
    """
    return score_compound_for_deployment(
        smiles,
        bundle["pipeline"],
        bundle["retained_descriptors"],
        bundle["threshold"],
        bundle["X_train_imp"],
        bundle["y_train"],
        bundle["oof_proba_train"],
        embed_fn=embed_only,
        minimize_fn=minimize_only,
        descriptor_fn=_mordred_fn,
        curate_fn=_curation_fn,
        descriptor_name_map=bundle["descriptor_name_map"],
        weights=weights,
    )


def score_many(smiles_input, bundle: dict, smiles_col: str = "SMILES",
               weights: tuple = (0.5, 0.0, 0.5), verbose: bool = True) -> pd.DataFrame:
    """Scores a batch: single SMILES, list/tuple, pandas Series/DataFrame,
    or a path to a .csv file. See score_compounds_for_deployment() in
    full_bbb_workflow.py for the full accepted-input list.
    """
    return score_compounds_for_deployment(
        smiles_input,
        bundle["pipeline"],
        bundle["retained_descriptors"],
        bundle["threshold"],
        bundle["X_train_imp"],
        bundle["y_train"],
        bundle["oof_proba_train"],
        embed_fn=embed_only,
        minimize_fn=minimize_only,
        descriptor_fn=_mordred_fn,
        curate_fn=_curation_fn,
        descriptor_name_map=bundle["descriptor_name_map"],
        weights=weights,
        smiles_col=smiles_col,
        verbose=verbose,
    )
