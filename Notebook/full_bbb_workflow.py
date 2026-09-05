"""
full_bbb_workflow.py

Implements the complete workflow, with an explicit config option at every
"/" branch point:

  1. MW filtering:          config["mw_mode"]        = "range" | "min_only"
  2. Train/test split:      config["split_method"]   = "random" | "scaffold"
                             (performed BEFORE feature selection and
                             winsorization so both are fit on the training
                             set only and frozen onto the test set, with no
                             leakage from test into either step)
  3. Feature selection:     distance correlation (>0.25) then Spearman
                             redundancy pruning (>0.85) -- single path,
                             thresholds configurable; fit on TRAIN only
  4. Winsorization:         config["winsorize"]      = True | False;
                             bounds fit on TRAIN only, frozen onto test
  5. Imputation comparison: config["imputers"]       = list from
                             {"mean","median","KNN","MICE","class_specific"}
  6. Fold-by-fold CV:       single-level CV -- the same cv folds are used for
                             BOTH grid search and performance reporting (no
                             separate inner_cv); calibrate (isotonic
                             regression, default; Platt/sigmoid still
                             selectable) per fold, no leakage
  7. Threshold selection:   exhaustive search maximizing MCC subject to
                             TNR(BBB-) >= TPR(BBB+)
  8. Evaluation:            MCC, G-mean, F1, AUPRC, Stratified Brier Score
                             (Wallace & Dahabreh, ICDM 2012, Eqs. 3-4),
                             calibration curve
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict

import dcor
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    matthews_corrcoef, f1_score, average_precision_score,
    confusion_matrix, brier_score_loss, roc_auc_score, roc_curve,
    accuracy_score, precision_score,
)

RANDOM_STATE = 42

# Models that are scale-sensitive and need standardized features (fit on
# training fold only, per fold -- see run_cv_imputation_model_comparison).
# Tree-based models split on raw thresholds and don't need this.
SCALE_SENSITIVE_MODELS = {"SVC", "LogisticRegression"}


# =========================================================================== #
# 1. MOLECULAR WEIGHT FILTERING
# =========================================================================== #

def filter_by_molecular_weight(
    df: pd.DataFrame, mw_col: str = "MW", mode: str = "range",
    low: float = 150.0, high: float = 800.0,
) -> pd.DataFrame:
    """
    mode="range"    -- keep only [low, high] (default 150-800 Da)
    mode="min_only" -- remove only compounds below `low`, no upper bound
    """
    if mode == "range":
        mask = (df[mw_col] >= low) & (df[mw_col] <= high)
    elif mode == "min_only":
        mask = df[mw_col] >= low
    else:
        raise ValueError("mode must be 'range' or 'min_only'")

    n_before = len(df)
    result = df[mask].reset_index(drop=True)
    print(f"MW filter ({mode}): {n_before} -> {len(result)} compounds "
          f"({n_before - len(result)} removed)")
    return result


# =========================================================================== #
# 2. FEATURE SELECTION: distance correlation, then Spearman redundancy pruning
# =========================================================================== #

def select_features_distance_correlation(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.25
) -> pd.Series:
    """
    Computes distance correlation (captures both linear and nonlinear
    association, unlike Pearson/Spearman) between each descriptor and the
    target. Returns a Series of distance-correlation values, sorted
    descending, for descriptors exceeding `threshold`.
    """
    scores = {}
    y_arr = y.values.astype(float)
    for col in X.columns:
        x_arr = X[col].values.astype(float)
        valid = ~np.isnan(x_arr)
        if valid.sum() < 10:  # not enough data to compute meaningfully
            continue
        try:
            scores[col] = dcor.distance_correlation(x_arr[valid], y_arr[valid])
        except Exception:
            continue

    scores = pd.Series(scores).sort_values(ascending=False)
    retained = scores[scores > threshold]
    print(f"Distance correlation: {len(retained)} / {len(X.columns)} descriptors "
          f"retained (threshold > {threshold})")
    return retained


def remove_redundant_features_spearman(
    X: pd.DataFrame, target_association: pd.Series, threshold: float = 0.85
) -> list[str]:
    """
    Computes pairwise Spearman correlation among the descriptors in
    target_association's index. For each pair exceeding `threshold`,
    keeps the one with STRONGER target association (from
    target_association, e.g. the distance correlation values from step 1)
    and discards its correlated partner.

    Returns the final list of retained descriptor names.
    """
    cols = target_association.index.tolist()
    corr_matrix = X[cols].corr(method="spearman").abs()

    to_drop = set()
    for i, col_i in enumerate(cols):
        if col_i in to_drop:
            continue
        for col_j in cols[i + 1:]:
            if col_j in to_drop:
                continue
            if corr_matrix.loc[col_i, col_j] > threshold:
                # drop whichever has WEAKER target association
                if target_association[col_i] >= target_association[col_j]:
                    to_drop.add(col_j)
                else:
                    to_drop.add(col_i)
                    break  # col_i is now dropped, move to next i

    retained = [c for c in cols if c not in to_drop]
    print(f"Spearman redundancy pruning: {len(cols)} -> {len(retained)} descriptors "
          f"({len(to_drop)} dropped for |rho| > {threshold})")
    return retained


# =========================================================================== #
# 3. WINSORIZATION (optional)
# =========================================================================== #

def winsorize_columns(
    X: pd.DataFrame, apply: bool = True, lower_pct: float = 0.01, upper_pct: float = 0.99
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    apply=True  -- winsorize each column at [lower_pct, upper_pct]
    apply=False -- pass through unchanged
    Returns (X_processed, bounds_df_or_None). bounds_df needed to apply the
    SAME frozen bounds to a held-out test set later.
    """
    if not apply:
        print("Winsorization: skipped (apply=False)")
        return X.copy(), None

    X_out = X.copy()
    bounds = {}
    skipped_bool_cols = []
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            # boolean/flag columns: winsorizing a 0/1 flag is meaningless, and
            # pandas' quantile interpolation errors on boolean dtype anyway
            # (numpy can't subtract booleans for the lerp step).
            skipped_bool_cols.append(col)
            continue
        lower = X[col].astype(float).quantile(lower_pct)
        upper = X[col].astype(float).quantile(upper_pct)
        X_out[col] = X[col].astype(float).clip(lower, upper)
        bounds[col] = {"lower_bound": lower, "upper_bound": upper}
    bounds_df = pd.DataFrame(bounds).T if bounds else None
    n_winsorized = len(X.columns) - len(skipped_bool_cols)
    print(f"Winsorization: applied at [{lower_pct}, {upper_pct}] to {n_winsorized} descriptors")
    if skipped_bool_cols:
        print(f"Winsorization: skipped {len(skipped_bool_cols)} boolean/flag column(s): {skipped_bool_cols}")
    return X_out, bounds_df


def apply_frozen_winsor_bounds(X: pd.DataFrame, bounds_df: pd.DataFrame | None) -> pd.DataFrame:
    if bounds_df is None:
        return X.copy()
    X_out = X.copy()
    for col in bounds_df.index:
        if col in X_out.columns:
            X_out[col] = X_out[col].clip(bounds_df.loc[col, "lower_bound"], bounds_df.loc[col, "upper_bound"])
    return X_out




# =========================================================================== #
# 4. TRAIN/TEST SPLIT: random or scaffold
# =========================================================================== #

def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    s = Chem.MolToSmiles(scaffold) if scaffold is not None else ""
    return s if s else "NO_RING"


def split_data(
    df: pd.DataFrame, method: str = "random", smiles_col: str = "Smiles",
    y_col: str | None = None, test_size: float = 0.15, random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    idx = np.arange(n)

    if method == "random":
        stratify = df[y_col].values if y_col else None
        return train_test_split(idx, test_size=test_size, stratify=stratify, random_state=random_state)

    elif method == "scaffold":
        scaffolds = df[smiles_col].apply(get_scaffold).values
        clusters = defaultdict(list)
        for i, s in enumerate(scaffolds):
            clusters[s].append(i)
        cluster_keys = list(clusters.keys())
        rng = np.random.RandomState(random_state)
        rng.shuffle(cluster_keys)

        if y_col is None:
            # No class labels available -- fall back to size-only assignment
            # (original behavior). Class balance cannot be checked without y.
            n_test_target = int(round(n * test_size))
            test_idx, train_idx = [], []
            running = 0
            for key in cluster_keys:
                members = clusters[key]
                (test_idx if running < n_test_target else train_idx).extend(members)
                running += len(members)
            return np.array(train_idx), np.array(test_idx)

        # Class-balance-aware assignment: scaffolds are still kept fully
        # intact (never split across train/test -- that's the whole point
        # of scaffold splitting, to test generalization to unseen chemistry),
        # but clusters are assigned greedily so each class's train/test
        # proportion stays close to its OVERALL proportion in the full
        # dataset -- not just the total row count. Without this, a single
        # large scaffold cluster that happens to be mostly one class (e.g.
        # BBB-, the minority class here) can silently starve train or test
        # of examples needed to learn/evaluate that class properly.
        y_arr = df[y_col].values
        classes = np.unique(y_arr)
        class_totals = {c: int((y_arr == c).sum()) for c in classes}
        test_target = {c: class_totals[c] * test_size for c in classes}

        # Largest clusters first: with quotas still fully open, big clusters
        # get placed where they help balance most; small clusters are left
        # to fine-tune the remainder at the end.
        cluster_keys.sort(key=lambda k: len(clusters[k]), reverse=True)

        test_idx, train_idx = [], []
        test_class_counts = {c: 0 for c in classes}
        n_test_target_total = n * test_size

        for key in cluster_keys:
            members = clusters[key]
            member_classes = y_arr[members]
            cluster_class_counts = {c: int((member_classes == c).sum()) for c in classes}

            # Would assigning this cluster to test push any class's test
            # count meaningfully past its target share? Score both options
            # by total squared deviation from each class's test target, and
            # take whichever is smaller -- this is what keeps BOTH classes'
            # train/test proportions close to the full dataset's, not just
            # the overall row count.
            test_if_added = sum(
                (test_class_counts[c] + cluster_class_counts[c] - test_target[c]) ** 2 for c in classes
            )
            test_if_skipped = sum(
                (test_class_counts[c] - test_target[c]) ** 2 for c in classes
            )
            # Softly discourage overshooting the total test_size target too,
            # not just per-class balance, once test is already near full.
            current_test_total = sum(test_class_counts.values())
            size_penalty = 1.0 if current_test_total < n_test_target_total else 1.5

            if test_if_added * size_penalty <= test_if_skipped:
                test_idx.extend(members)
                for c in classes:
                    test_class_counts[c] += cluster_class_counts[c]
            else:
                train_idx.extend(members)

        return np.array(train_idx), np.array(test_idx)

    raise ValueError("method must be 'random' or 'scaffold'")


# =========================================================================== #
# 5. IMPUTATION STRATEGIES
# =========================================================================== #

def get_imputer(name: str, random_state: int = RANDOM_STATE):
    if name == "mean":
        return SimpleImputer(strategy="mean")
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "KNN":
        return KNNImputer(n_neighbors=5)
    if name == "MICE":
        return IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=random_state)
    raise ValueError(f"Unknown imputer: {name}")


def class_specific_impute_fit_apply(X_train, y_train, X_apply, strategy="median"):
    """
    Fits per-class statistic (median or mean) on X_train/y_train, applies
    to X_apply. NOTE: at true deployment the class is unknown, so this is
    a training-time comparison baseline only (flagged since the start of
    this pipeline) -- falls back to the overall statistic when applying,
    since a real query has no class label.
    """
    if strategy == "median":
        overall = X_train.median()
    else:
        overall = X_train.mean()
    X_out = X_apply.copy()
    for col in X_out.columns:
        X_out[col] = X_out[col].fillna(overall[col])
    return X_out


# =========================================================================== #
# 5b. MODEL REGISTRY: estimator + grid-search hyperparameter space per model
# =========================================================================== #

def get_model_and_grid(name: str, random_state: int = RANDOM_STATE):
    """
    Returns (unfitted_estimator, param_grid) for GridSearchCV.
    Grids are kept deliberately small -- nested CV (grid search inside every
    outer fold, for every imputer, for every model) is already expensive;
    widen these once the pipeline is confirmed to run end-to-end.

    Every model that can parallelize internally (RandomForest, XGBoost,
    CatBoost, LightGBM) is pinned to a SINGLE thread here. GridSearchCV is
    already given n_jobs=-1 in run_cv_imputation_model_comparison, which
    parallelizes across hyperparameter combinations/folds itself -- if each
    individual model ALSO tries to use all cores, you get oversubscription
    (many threads competing for the same cores), which slows things down
    rather than speeding them up. Letting GridSearchCV be the only thing
    doing the multithreading is what actually uses multi-core machines (like
    System 2) efficiently.
    """
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=random_state, n_jobs=1)
        grid = {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]}

    elif name == "SVC":
        from sklearn.svm import SVC
        # probability=False: CalibratedClassifierCV re-derives probabilities
        # from decision_function itself, so SVC's own (slower, nested-CV)
        # probability estimation would be redundant here.
        # (SVC has no internal multithreading knob to pin -- it's single-
        # threaded by nature, so nothing to change here.)
        #
        # max_iter=10000 (sklearn's default is -1 = unlimited): certain
        # (C, gamma) combinations, especially C=10 with gamma='auto' on
        # unscaled/imperfectly-scaled data, can converge extremely slowly or
        # effectively hang. Capping iterations means a single bad combination
        # gets cut off with a (harmless) non-convergence warning rather than
        # silently stalling the entire nested CV run for hours.
        model = SVC(probability=False, random_state=random_state, max_iter=10000)
        grid = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}

    elif name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        # LogisticRegression's default solver ('lbfgs') is single-threaded
        # for a single fit already -- nothing to pin here either.
        model = LogisticRegression(max_iter=2000, random_state=random_state)
        grid = {"C": [0.01, 0.1, 1, 10]}

    elif name == "XGBoost":
        from xgboost import XGBClassifier
        model = XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=1)
        grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}

    elif name == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            verbose=0, random_state=random_state, allow_writing_files=False, thread_count=1,
        )
        grid = {"depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "iterations": [200, 400]}

    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=1)
        grid = {"n_estimators": [200, 400], "max_depth": [-1, 5, 10], "learning_rate": [0.05, 0.1]}

    else:
        raise ValueError(
            f"Unknown model: {name!r}. Expected one of: "
            "RandomForest, SVC, LogisticRegression, XGBoost, CatBoost, LightGBM"
        )
    return model, grid



def stratified_brier_score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Wallace & Dahabreh (ICDM 2012), Eqs. 2-4."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pos, neg = y_true == 1, y_true == 0
    return {
        "BS_overall": float(np.mean((y_true - proba) ** 2)),
        "BS_positive": float(np.mean((y_true[pos] - proba[pos]) ** 2)) if pos.any() else np.nan,
        "BS_negative": float(np.mean((y_true[neg] - proba[neg]) ** 2)) if neg.any() else np.nan,
    }


def g_mean_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(sensitivity * specificity))


def find_constrained_threshold(
    y_true: np.ndarray, proba: np.ndarray, metric: str = "MCC",
    constraint: str = "TNR>=TPR",
) -> dict:
    """
    Exhaustive search over observed probability values: maximize `metric`
    subject to the constraint TNR(BBB-) >= TPR(BBB+) (or none).
    """
    y_true = np.asarray(y_true)
    thresholds = np.unique(proba)
    best_t, best_score = 0.5, -np.inf
    satisfied_any = False

    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0

        if constraint == "TNR>=TPR" and tnr < tpr:
            continue
        satisfied_any = True

        score = matthews_corrcoef(y_true, pred) if metric == "MCC" else f1_score(y_true, pred)
        if score > best_score:
            best_score, best_t = score, t

    if not satisfied_any:
        # no threshold satisfies the constraint -- fall back to unconstrained best
        best_t, best_score = find_constrained_threshold(y_true, proba, metric, constraint=None)["threshold"], None
        return {"threshold": best_t, "score": best_score, "constraint_satisfied": False}

    return {"threshold": best_t, "score": best_score, "constraint_satisfied": True}



# =========================================================================== #
# FULL METRIC SUITE -- for train (CV out-of-fold), test, and external sets,
# all via the SAME function so numbers are directly comparable across sets.
# Adds AUC-ROC, Accuracy, Precision, Sensitivity, Specificity to the metrics
# already used during CV (MCC, G-mean, F1, AUPRC, stratified Brier).
# =========================================================================== #

def evaluate_predictions(y_true, y_pred, y_proba) -> pd.Series:
    """
    Full metric report for one set of predictions with known true labels.
    Use identically for the CV out-of-fold set, the held-out test set, and
    an external labeled set -- same function, same metric definitions,
    so the three are directly comparable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    bs = stratified_brier_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # = TPR = recall (BBB+)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # = TNR (BBB-)

    return pd.Series({
        "MCC": matthews_corrcoef(y_true, y_pred),
        "G_mean": g_mean_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUPRC": average_precision_score(y_true, y_proba),
        "AUC_ROC": roc_auc_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "BS_overall": bs["BS_overall"],
        "BS_positive": bs["BS_positive"],
        "BS_negative": bs["BS_negative"],
    })


# =========================================================================== #
# FINAL MODEL FITTING + PREDICTION ON TEST / EXTERNAL SETS
# =========================================================================== #

def fit_final_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series, imputer_name: str, model_name: str,
    best_params: dict, calibration_method: str = "isotonic", calibration_cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Fits ONE final pipeline on the FULL training set (not a CV fold) --
    imputer, then scaler if this model needs one, then the calibrated
    model. This is what gets applied to the test set and external set.
    """
    from sklearn.base import clone

    if imputer_name == "class_specific":
        X_train_imp = class_specific_impute_fit_apply(X_train, y_train, X_train)
        fitted_imputer = None
    else:
        fitted_imputer = get_imputer(imputer_name, random_state)
        X_train_imp = pd.DataFrame(
            fitted_imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns
        )

    fitted_scaler = None
    if model_name in SCALE_SENSITIVE_MODELS:
        fitted_scaler = StandardScaler()
        X_train_imp = pd.DataFrame(
            fitted_scaler.fit_transform(X_train_imp), index=X_train_imp.index, columns=X_train_imp.columns
        )

    base_model, _ = get_model_and_grid(model_name, random_state)
    tuned_model = clone(base_model).set_params(**best_params)
    calibrated = CalibratedClassifierCV(tuned_model, method=calibration_method, cv=calibration_cv)
    calibrated.fit(X_train_imp, y_train)

    return {
        "imputer": fitted_imputer, "scaler": fitted_scaler, "model": calibrated,
        "imp_name": imputer_name, "model_name": model_name,
        "class_specific_reference": (X_train, y_train) if imputer_name == "class_specific" else None,
    }


def transform_with_pipeline(pipeline: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Applies a fit_final_pipeline()'s imputer+scaler to new data (test/external set).
    ONLY .transform() is ever called here -- the imputer/scaler were fit
    exclusively on X_train, so nothing about the test/external set's own
    values influences how it gets imputed/scaled. No leakage."""
    if pipeline["imp_name"] == "class_specific":
        X_tr_ref, y_tr_ref = pipeline["class_specific_reference"]
        X_imp = class_specific_impute_fit_apply(X_tr_ref, y_tr_ref, X)
    else:
        X_imp = pd.DataFrame(pipeline["imputer"].transform(X), index=X.index, columns=X.columns)

    if pipeline["scaler"] is not None:
        X_imp = pd.DataFrame(pipeline["scaler"].transform(X_imp), index=X_imp.index, columns=X_imp.columns)

    return X_imp


def align_external_descriptors(
    X_external_raw: pd.DataFrame, retained_descriptors: list, descriptor_name_map: dict | None = None,
    on_missing: str = "warn",
) -> pd.DataFrame:
    """
    THIS is the answer to "how do test/external sets, which have ALL
    descriptors, get reduced to just the ones selected during feature
    selection on the training set, before prediction."

    Two things have to be replicated from training, not just a plain
    column subset:

      1. Column NAME sanitization: training's descriptor columns were
         renamed (special JSON characters stripped, for LightGBM
         compatibility -- e.g. "fr_C(=O)O" became "fr_C_O_O") BEFORE
         feature selection ran. So `retained_descriptors` contains
         SANITIZED names, but X_external_raw (freshly computed via your
         normal descriptor pipeline) will have the ORIGINAL names. This
         function sanitizes X_external_raw's columns the SAME way first.

      2. Column SELECTION + ORDER: the external/test set has EVERY
         descriptor computed, but the model only uses the handful that
         survived distance-correlation + Spearman pruning on TRAINING.
         This subsets AND reorders to exactly `retained_descriptors`,
         since sklearn/LightGBM are positional once inside .predict().

    Imputation itself does NOT happen here -- it happens afterward, inside
    predict_on_new_data() below, using the imputer that was fit ONLY on
    X_train (see transform_with_pipeline). This function's job is just
    getting the right COLUMNS in the right ORDER; missing values in those
    columns are left as NaN here and filled by the trained imputer next.

    on_missing: "warn" (default, fills missing descriptors with NaN and
      prints which ones, by original name) | "error" | "ignore"
    """
    X_sanitized, _ = sanitize_column_names(X_external_raw)
    name_map = descriptor_name_map or {}

    missing = [c for c in retained_descriptors if c not in X_sanitized.columns]
    if missing:
        missing_original = [name_map.get(c, c) for c in missing]
        msg = (f"{len(missing)} of {len(retained_descriptors)} expected descriptor(s) not found "
               f"in this dataset (original names): {missing_original}")
        if on_missing == "error":
            raise ValueError(msg)
        elif on_missing == "warn":
            print(f"WARNING: {msg}\n  -> filling with NaN; the trained imputer will handle these "
                  f"as missing values, same as during training.")
        for c in missing:
            X_sanitized[c] = float("nan")

    extra = set(X_sanitized.columns) - set(retained_descriptors)
    if extra:
        print(f"Dropping {len(extra)} descriptor(s) present here but not used by the trained "
              f"model (not part of the feature-selected set).")

    return X_sanitized[retained_descriptors]


def predict_on_new_data(
    pipeline: dict, X_new_raw: pd.DataFrame, retained_descriptors: list, threshold: float,
    descriptor_name_map: dict | None = None, y_true=None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Complete path for scoring a NEW dataset (test set or external set) with
    the final trained pipeline: align descriptors -> impute/scale (fit on
    X_train only) -> calibrated predict_proba -> apply the threshold.

    If y_true is given (known labels -- e.g. your held-out test set, or an
    external set you have labels for), also returns the full metric suite
    via evaluate_predictions(), so results are directly comparable to the
    CV numbers using the exact same metric definitions.

    Returns (predictions_df, metrics_or_None).
    predictions_df has columns [probability, prediction].
    """
    X_aligned = align_external_descriptors(X_new_raw, retained_descriptors, descriptor_name_map)
    X_imp = transform_with_pipeline(pipeline, X_aligned)

    proba = pipeline["model"].predict_proba(X_imp)[:, 1]
    pred = (proba >= threshold).astype(int)
    predictions_df = pd.DataFrame({"probability": proba, "prediction": pred}, index=X_new_raw.index)

    metrics = evaluate_predictions(y_true, pred, proba) if y_true is not None else None
    return predictions_df, metrics


# =========================================================================== #
# PREDICTION RELIABILITY SCORING
# Adapted from: Roy, Ambure & Kar, "How Precise Are Our QSAR-Derived
# Predictions for New Query Chemicals?", ACS Omega 2018, 3, 11392-11406.
# DOI: 10.1021/acsomega.8b01647
#
# The original scheme was built for MLR REGRESSION models. Adapting it to a
# calibrated BINARY CLASSIFIER requires three substitutions, each noted
# below at the relevant rule.
# =========================================================================== #

def get_oof_probabilities_for_final_params(
    X_train_imp: pd.DataFrame, y_train: pd.Series, model_name: str, best_params: dict,
    calibration_method: str = "isotonic", calibration_cv: int = 5, cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Out-of-fold calibrated probabilities for EVERY training compound, using
    the exact final hyperparameters already chosen -- this is the practical
    stand-in for Rule 1's true leave-one-out predictions (true LOO would
    mean retraining once per training compound, prohibitive at this
    dataset size). k-fold OOF is the standard, accepted approximation.
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = np.full(len(X_train_imp), np.nan)
    base_model, _ = get_model_and_grid(model_name, random_state)

    for train_idx, val_idx in skf.split(X_train_imp, y_train):
        X_tr, X_val = X_train_imp.iloc[train_idx], X_train_imp.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        tuned_model = clone(base_model).set_params(**best_params)
        calibrated = CalibratedClassifierCV(tuned_model, method=calibration_method, cv=calibration_cv)
        calibrated.fit(X_tr, y_tr)
        oof_proba[val_idx] = calibrated.predict_proba(X_val)[:, 1]

    return oof_proba


def compute_reliability_scores(
    X_train_imp: pd.DataFrame, y_train: pd.Series, oof_proba_train: np.ndarray,
    X_query_imp: pd.DataFrame, query_proba: np.ndarray,
    k_neighbors: int = 10, ed_outlier_k: float = 3.0, weights: tuple = (0.5, 0.0, 0.5),
    max_train_pairs_for_ed_threshold: int = 3000,
) -> pd.DataFrame:
    """
    Composite prediction-reliability score (Roy et al. 2018), adapted for a
    binary classifier:
      Rule 1 (LOO error of 10 nearest neighbors): uses k-fold OUT-OF-FOLD
        probabilities in place of true LOO. "Response range" = 1 (binary
        target), so thresholds are fixed (0.1/0.15 for MAE, 0.2/0.25 for
        MAE+3sigma) rather than scaled by a continuous training range.
      Rule 2 (AD via standardization): unchanged from the original.
      Rule 3 (proximity to training response mean): "training response
        mean" = class prevalence (Bernoulli), compared against the query's
        PREDICTED PROBABILITY.
    Composite = w1*rule1 + w2*rule2 + w3*rule3, rounded, clipped to [1,3].
    Default weights (0.5, 0.0, 0.5) match the paper's own most frequently
    optimal weighting.
    """
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_imp)
    X_query_std = scaler.transform(X_query_imp)
    y_train_arr = y_train.values

    n_train = len(X_train_std)
    rng = np.random.RandomState(RANDOM_STATE)
    if n_train > max_train_pairs_for_ed_threshold:
        idx_sample = rng.choice(n_train, size=max_train_pairs_for_ed_threshold, replace=False)
        sample = X_train_std[idx_sample]
    else:
        sample = X_train_std
    from scipy.spatial.distance import pdist
    pairwise_ed = pdist(sample, metric="euclidean")
    ed_threshold = pairwise_ed.mean() + ed_outlier_k * pairwise_ed.std()

    y_mean, y_std = y_train_arr.mean(), y_train_arr.std()

    rows = []
    for i in range(len(X_query_std)):
        q_std = X_query_std[i]
        q_proba = query_proba[i]

        dists = np.linalg.norm(X_train_std - q_std, axis=1)
        nearest_idx = np.argsort(dists)[:k_neighbors]
        nearest_idx = nearest_idx[dists[nearest_idx] <= ed_threshold]

        if len(nearest_idx) < 3:
            rule1_score = 1
            mae_loo, n_neighbors_used = np.nan, len(nearest_idx)
        else:
            abs_errors = np.abs(y_train_arr[nearest_idx] - oof_proba_train[nearest_idx])
            mae_loo, std_loo = abs_errors.mean(), abs_errors.std()
            n_neighbors_used = len(nearest_idx)
            if mae_loo <= 0.1 and (mae_loo + 3 * std_loo) <= 0.2:
                rule1_score = 3
            elif mae_loo > 0.15 or (mae_loo + 3 * std_loo) > 0.25:
                rule1_score = 1
            else:
                rule1_score = 2

        abs_s = np.abs(q_std)
        if abs_s.max() < 3:
            rule2_score = 3
        elif abs_s.min() > 3:
            rule2_score = 1
        else:
            s_new = abs_s.mean() + 1.28 * abs_s.std()
            rule2_score = 2 if s_new < 3 else 1

        if (y_mean - 2 * y_std) <= q_proba <= (y_mean + 2 * y_std):
            rule3_score = 3
        elif (y_mean - 3 * y_std) <= q_proba <= (y_mean + 3 * y_std):
            rule3_score = 2
        else:
            rule3_score = 1

        composite_raw = weights[0] * rule1_score + weights[1] * rule2_score + weights[2] * rule3_score
        composite_score = int(np.clip(round(composite_raw), 1, 3))

        rows.append({
            "rule1_score": rule1_score, "rule2_score": rule2_score, "rule3_score": rule3_score,
            "composite_score": composite_score,
            "MAE_LOO": mae_loo, "n_neighbors_used": n_neighbors_used,
        })

    return pd.DataFrame(rows, index=X_query_imp.index)


def compute_threshold_margin_score(
    query_proba: np.ndarray, threshold: float, oof_proba_train: np.ndarray,
) -> np.ndarray:
    """
    A prediction sitting right next to the decision threshold is inherently
    borderline -- a probability of 0.67 vs. the 0.66 threshold is barely
    distinguishable from 0.64, yet one gets labeled BBB+ and the other
    BBB-. The Roy et al. rules above don't capture this at all -- they
    judge structural/statistical representativeness, not decision-boundary
    proximity. This adds that as a SEPARATE score.

    Margin = |query_proba - threshold|. Scored against the empirical
    distribution of |oof_proba_train - threshold| (data-driven, not an
    arbitrary fixed cutoff): top tertile of margin -> 3 (clearly on one
    side), middle tertile -> 2, bottom tertile (closest to the threshold,
    most borderline) -> 1.
    """
    train_margins = np.abs(oof_proba_train - threshold)
    tertile_33, tertile_67 = np.percentile(train_margins, [33.33, 66.67])

    query_margins = np.abs(np.asarray(query_proba) - threshold)
    scores = np.where(query_margins >= tertile_67, 3, np.where(query_margins >= tertile_33, 2, 1))
    return scores


def predict_bbb_with_confidence(
    pipeline: dict, X_new_raw: pd.DataFrame, retained_descriptors: list, threshold: float,
    X_train_imp: pd.DataFrame, y_train: pd.Series, oof_proba_train: np.ndarray,
    descriptor_name_map: dict | None = None, weights: tuple = (0.5, 0.0, 0.5),
) -> pd.DataFrame:
    """
    Full prediction + combined confidence path. FINAL confidence =
    min(Roy et al. composite_score, threshold_margin_score) -- conservative
    by design: a prediction is only as trustworthy as the WEAKER of "is
    this compound well-represented by training data" and "is this
    prediction comfortably on one side of the decision threshold." Either
    signal alone flagging concern is enough to downgrade the label.

    Returns predictions_df with columns: probability, prediction,
    composite_score (Roy et al.), threshold_margin_score,
    final_confidence_score, final_confidence_label.
    """
    X_aligned = align_external_descriptors(X_new_raw, retained_descriptors, descriptor_name_map)
    X_imp = transform_with_pipeline(pipeline, X_aligned)

    proba = pipeline["model"].predict_proba(X_imp)[:, 1]
    pred = (proba >= threshold).astype(int)

    reliability = compute_reliability_scores(X_train_imp, y_train, oof_proba_train, X_imp, proba, weights=weights)
    margin_scores = compute_threshold_margin_score(proba, threshold, oof_proba_train)

    final_score = np.minimum(reliability["composite_score"].values, margin_scores)
    final_label = np.select(
        [final_score == 3, final_score == 2, final_score == 1],
        ["Good", "Moderate", "Poor/Unreliable"],
        default="Unknown",
    )

    result = pd.DataFrame({
        "probability": proba, "prediction": pred,
        "composite_score": reliability["composite_score"].values,
        "threshold_margin_score": margin_scores,
        "final_confidence_score": final_score,
        "final_confidence_label": final_label,
    }, index=X_new_raw.index)

    return result


# =========================================================================== #
# FULL DEPLOYMENT SCORER -- one row per compound, exactly the columns
# needed for the deployed server output: 3D-generation status/reason,
# prediction, confidence, BBB+ probability (%), and success/failure status.
# =========================================================================== #

def score_compound_for_deployment(
    smiles: str, pipeline: dict, retained_descriptors: list, threshold: float,
    X_train_imp: pd.DataFrame, y_train: pd.Series, oof_proba_train: np.ndarray,
    embed_fn, minimize_fn, descriptor_fn, curate_fn=None,
    descriptor_name_map: dict | None = None, weights: tuple = (0.5, 0.0, 0.5),
) -> dict:
    """
    Full per-compound deployment path: curation -> 3D generation -> force-
    field minimization -> descriptor calculation -> prediction + combined
    confidence, into ONE row with exactly the columns needed for server
    output.

    curate_fn, embed_fn, minimize_fn, descriptor_fn are all INJECTED (not
    imported directly) so this stays decoupled from your BBB.py/Mordred
    setup:

      curate_fn(smiles) -> (curated_smiles_or_None, curation_status)
        Wire in your BBB.py curation step here (canonicalize_smiles +
        final_curated_df's per-compound logic -- salt/counterion
        stripping, exclusion of simple inorganics, validity checks).
        This is the stage BBB.py runs BEFORE 3D embedding -- skipping it
        means a raw, uncurated SMILES (possibly a salt form, non-
        canonical, or something that should be excluded) goes straight
        into embed_fn. curate_fn is OPTIONAL but strongly recommended:
        if you don't pass one, curation is skipped and flagged as such
        in Curation_Status, rather than silently assumed already clean.

      embed_fn(smiles) -> (mol_or_None, embed_status)
        Your BBB.py 3D-generation ladder (embed_only).

      minimize_fn(mol) -> (mol_or_None, minimize_status)
        Your BBB.py force-field step (minimize_only).

      descriptor_fn(mol) -> single-row DataFrame of raw (unsanitized,
        unselected) descriptor columns -- your Mordred wrapper.

    Returns a dict with keys: SMILES, Curation_Status, 3D_Generation_Status,
    Prediction, Confidence, BBB_plus_Probability_Percent, Status.
    """
    row = {"SMILES": smiles, "Curation_Status": None, "3D_Generation_Status": None,
           "Prediction": None, "Confidence": None, "BBB_plus_Probability_Percent": None,
           "Status": "Failed"}

    # --- Stage 0: curation (BBB.py's pre-embedding step) ---
    if curate_fn is not None:
        curated_smiles, curation_status = curate_fn(smiles)
        row["Curation_Status"] = curation_status
        if curated_smiles is None:
            row["3D_Generation_Status"] = "not_attempted"
            return row
    else:
        curated_smiles = smiles
        row["Curation_Status"] = "skipped_no_curate_fn"

    # --- Stage 1: 3D conformer generation (your BBB.py ladder) ---
    mol, embed_status = embed_fn(curated_smiles)
    row["3D_Generation_Status"] = embed_status

    if mol is None or embed_status == "ok_chirality_unenforced":
        # invalid_smiles / not_strained_no_ladder / embedding_failed:... /
        # OR only the chirality-unenforced stage succeeded -- excluded per
        # your study's design (ORCA never initiated on these).
        return row

    # --- Stage 2: force-field minimization ---
    mol_final, minimize_status = minimize_fn(mol)
    if mol_final is None or minimize_status != "ok":
        row["3D_Generation_Status"] = f"{embed_status} -> minimize_failed:{minimize_status}"
        return row

    # --- Stage 3: descriptor calculation ---
    try:
        X_new_raw = descriptor_fn(mol_final)
    except Exception as e:
        row["3D_Generation_Status"] = f"{embed_status} -> descriptor_calc_failed:{e}"
        return row

    # --- Stage 4: prediction + combined confidence ---
    try:
        result = predict_bbb_with_confidence(
            pipeline, X_new_raw, retained_descriptors, threshold,
            X_train_imp, y_train, oof_proba_train,
            descriptor_name_map=descriptor_name_map, weights=weights,
        )
        r = result.iloc[0]
        row["Prediction"] = "BBB+" if r["prediction"] == 1 else "BBB-"
        row["Confidence"] = r["final_confidence_label"]
        row["BBB_plus_Probability_Percent"] = round(r["probability"] * 100, 2)
        row["Status"] = "Success"
    except Exception as e:
        row["3D_Generation_Status"] = f"{embed_status} -> prediction_failed:{e}"

    return row


def score_compounds_for_deployment(
    smiles_input, pipeline: dict, retained_descriptors: list, threshold: float,
    X_train_imp: pd.DataFrame, y_train: pd.Series, oof_proba_train: np.ndarray,
    embed_fn, minimize_fn, descriptor_fn, curate_fn=None,
    descriptor_name_map: dict | None = None, weights: tuple = (0.5, 0.0, 0.5),
    smiles_col: str = "SMILES", verbose: bool = True,
) -> pd.DataFrame:
    """
    Same full deployment path as score_compound_for_deployment(), but
    accepts EITHER a single SMILES OR a batch, and always returns a
    DataFrame (one row per compound, same columns as the single-compound
    version) -- this is what you'd actually call from a CSV-upload
    endpoint or a bulk-scoring script, rather than looping
    score_compound_for_deployment() yourself.

    smiles_input can be:
      - a single SMILES string          -> "CCO"
      - a list/tuple of SMILES strings   -> ["CCO", "c1ccccc1O", ...]
      - a pandas Series of SMILES        -> df["SMILES"]
      - a pandas DataFrame               -> uses the column named
                                             `smiles_col` (default "SMILES")
      - a path to a CSV file (string ending in .csv) -> reads it, then
                                             uses the `smiles_col` column

    Prints progress (compound N/total) if verbose=True, since 3D
    generation can be slow enough per-compound that silent looping over a
    large CSV would look stalled.
    """
    if isinstance(smiles_input, str) and smiles_input.lower().endswith(".csv"):
        smiles_list = pd.read_csv(smiles_input)[smiles_col].tolist()
    elif isinstance(smiles_input, str):
        smiles_list = [smiles_input]  # single SMILES
    elif isinstance(smiles_input, pd.DataFrame):
        smiles_list = smiles_input[smiles_col].tolist()
    elif isinstance(smiles_input, pd.Series):
        smiles_list = smiles_input.tolist()
    else:
        smiles_list = list(smiles_input)  # list/tuple/array of SMILES

    rows = []
    for i, smi in enumerate(smiles_list):
        if verbose:
            print(f"[{i + 1}/{len(smiles_list)}] Scoring: {smi}")
        row = score_compound_for_deployment(
            smi, pipeline, retained_descriptors, threshold,
            X_train_imp, y_train, oof_proba_train,
            embed_fn=embed_fn, minimize_fn=minimize_fn, descriptor_fn=descriptor_fn,
            curate_fn=curate_fn, descriptor_name_map=descriptor_name_map, weights=weights,
        )
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if verbose:
        print(f"\nDone: {(result_df['Status'] == 'Success').sum()}/{len(result_df)} succeeded")
    return result_df




def run_cv_imputation_model_comparison(
    X: pd.DataFrame, y: pd.Series, imputer_names: list[str], model_names: list[str],
    cv: int = 5, calibration_method: str = "isotonic", calibration_cv: int = 5,
    random_state: int = RANDOM_STATE, n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict]:
    """
    SINGLE-LEVEL cross-validation: the SAME `cv` outer folds are used for
    BOTH hyperparameter search and final performance reporting -- there is
    no separate inner_cv anymore.

    For every (imputer, model) combination:
      1. Impute each of the `cv` folds ONCE (fit on that fold's training
         rows only, no leakage). These fold-wise imputed (and, for
         SVC/LogisticRegression, scaled) datasets are then reused for both
         steps below -- computed once, not recomputed per hyperparameter
         candidate.
      2. For every hyperparameter combination in the model's grid: fit on
         each fold's training data, score (MCC) on that fold's held-out
         data, and average the score across all `cv` folds. That average
         IS this combination's cross-validated score -- computed directly
         from the same folds used for final evaluation, not a separate
         inner split. The single best-scoring combination is kept.
      3. Using that one winning combination, calibrate (isotonic by
         default -- see below) per fold on the SAME folds, predict on each
         fold's held-out rows, and report every metric (MCC, G-mean, F1,
         AUPRC, Brier) as mean +/- std across those `cv` folds.

    Trade-off, stated plainly: because hyperparameter selection uses the
    mean score across the SAME folds later used to report performance,
    this is single-level CV, not nested. It is simpler and roughly
    `n_grid_combinations` times cheaper than nested CV (no repeated inner
    search per outer fold), at the cost of a mild optimistic bias: each
    fold's held-out data contributed, in aggregate via the mean across
    folds, to choosing the hyperparameters later evaluated on that same
    fold. This is a common, standard simplification -- just not the
    zero-bias gold standard that fully nested CV would give.

    calibration_method="isotonic" (default): fits a non-parametric,
    monotonic step-function mapping from raw score -> calibrated probability,
    rather than assuming the sigmoid (Platt) shape. Preferred here over
    "sigmoid" because:
      - Platt/sigmoid assumes miscalibration is a specific S-shaped
        distortion; isotonic makes no such assumption, so it can correct
        whatever irregular miscalibration shape imbalanced classifiers
        actually produce (which is often NOT a clean sigmoid).
      - Isotonic needs more data per calibration fold to avoid overfitting
        its step function -- with ~7192 compounds total, there is enough
        data per fold for isotonic to be the more reliable choice.
      - Minority-class (BBB-) probabilities are exactly where a rigid
        sigmoid shape is most likely to misfit under imbalance; isotonic's
        flexibility matters most in that region.
    Pass calibration_method="sigmoid" to fall back to Platt scaling if a
    given fold is too small for isotonic to behave well.

    Threshold selection uses the POOLED out-of-fold probabilities across
    all folds (more data -> a more stable threshold than picking one per
    fold). Everything downstream of that is computed PER fold using that
    shared threshold, then reported as `<metric>_mean` / `<metric>_std`.

    Returns:
      comparison_df -- one row per (imputer, model), indexed by both, with
        columns threshold, constraint_satisfied, best_params, and
        <MCC/G_mean/F1/AUPRC/BS_overall/BS_positive/BS_negative>_mean/_std
      oof_proba_dict -- {"imputer|model": oof_proba_array} for calibration plots
    """
    from sklearn.base import clone
    from sklearn.model_selection import ParameterGrid
    from joblib import Parallel, delayed

    def _fit_score_fold(base_model, candidate, X_tr, y_tr, X_val, y_val):
        m = clone(base_model).set_params(**candidate)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        return matthews_corrcoef(y_val, pred)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    oof_proba_dict = {}

    for imp_name in imputer_names:
        # Impute every fold ONCE per imputer -- reused below for every model
        # and every hyperparameter candidate, rather than re-imputing per
        # candidate (which would be wasteful and wouldn't change anyway).
        fold_data = []  # (train_idx, val_idx, X_tr_imp, y_tr, X_val_imp, y_val)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if imp_name == "class_specific":
                X_tr_imp = class_specific_impute_fit_apply(X_tr, y_tr, X_tr)
                X_val_imp = class_specific_impute_fit_apply(X_tr, y_tr, X_val)
            else:
                imputer = get_imputer(imp_name, random_state)
                X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
                X_val_imp = pd.DataFrame(imputer.transform(X_val), index=X_val.index, columns=X_val.columns)
            fold_data.append((train_idx, val_idx, X_tr_imp, y_tr, X_val_imp, y_val))

        for model_name in model_names:
            base_model, param_grid = get_model_and_grid(model_name, random_state)

            # Fold-wise scaling for SVC/LogisticRegression (fit on that
            # fold's training portion only) -- same no-leakage rule as
            # imputation, computed once here and reused for every candidate.
            scaled_fold_data = []
            for (train_idx, val_idx, X_tr_imp, y_tr, X_val_imp, y_val) in fold_data:
                if model_name in SCALE_SENSITIVE_MODELS:
                    scaler = StandardScaler()
                    X_tr_s = pd.DataFrame(
                        scaler.fit_transform(X_tr_imp), index=X_tr_imp.index, columns=X_tr_imp.columns
                    )
                    X_val_s = pd.DataFrame(
                        scaler.transform(X_val_imp), index=X_val_imp.index, columns=X_val_imp.columns
                    )
                else:
                    X_tr_s, X_val_s = X_tr_imp, X_val_imp
                scaled_fold_data.append((X_tr_s, y_tr, X_val_s, val_idx, y_val))

            # Score every hyperparameter candidate on the SAME `cv` folds
            # that will later be used for evaluation -- this IS the
            # "no separate inner_cv" step: one round of scoring across the
            # outer folds picks the single best combination.
            candidates = list(ParameterGrid(param_grid))
            tasks = [(ci, fi) for ci in range(len(candidates)) for fi in range(len(scaled_fold_data))]
            flat_scores = Parallel(n_jobs=n_jobs)(
                delayed(_fit_score_fold)(
                    base_model, candidates[ci],
                    scaled_fold_data[fi][0], scaled_fold_data[fi][1],
                    scaled_fold_data[fi][2], scaled_fold_data[fi][4],
                )
                for (ci, fi) in tasks
            )
            scores_by_candidate = np.array(flat_scores).reshape(len(candidates), len(scaled_fold_data))
            mean_scores = scores_by_candidate.mean(axis=1)
            best_params = candidates[int(np.argmax(mean_scores))]

            # Calibrate + evaluate per fold using the ONE winning combination
            # -- same folds used above, so grid search and performance both
            # come from one cv set, as requested.
            oof_proba = np.full(len(X), np.nan)
            for (X_tr_s, y_tr, X_val_s, val_idx, y_val) in scaled_fold_data:
                tuned_model = clone(base_model).set_params(**best_params)
                calibrated = CalibratedClassifierCV(tuned_model, method=calibration_method, cv=calibration_cv)
                calibrated.fit(X_tr_s, y_tr)
                oof_proba[val_idx] = calibrated.predict_proba(X_val_s)[:, 1]

            key = f"{imp_name}|{model_name}"
            oof_proba_dict[key] = oof_proba

            # Threshold chosen ONCE from pooled out-of-fold probabilities
            # (more data -> more stable than picking one per fold).
            threshold_result = find_constrained_threshold(y.values, oof_proba, metric="MCC", constraint="TNR>=TPR")
            threshold = threshold_result["threshold"]

            # Metrics computed PER fold using that shared threshold, then
            # reported as mean +/- std across the `cv` folds.
            fold_metric_values = {m: [] for m in
                                   ["MCC", "G_mean", "F1", "AUPRC", "BS_overall", "BS_positive", "BS_negative"]}
            for (_, _, _, val_idx, y_val) in scaled_fold_data:
                proba_fold = oof_proba[val_idx]
                pred_fold = (proba_fold >= threshold).astype(int)
                bs_fold = stratified_brier_score(y_val.values, proba_fold)

                fold_metric_values["MCC"].append(matthews_corrcoef(y_val, pred_fold))
                fold_metric_values["G_mean"].append(g_mean_score(y_val, pred_fold))
                fold_metric_values["F1"].append(f1_score(y_val, pred_fold))
                fold_metric_values["AUPRC"].append(average_precision_score(y_val, proba_fold))
                fold_metric_values["BS_overall"].append(bs_fold["BS_overall"])
                fold_metric_values["BS_positive"].append(bs_fold["BS_positive"])
                fold_metric_values["BS_negative"].append(bs_fold["BS_negative"])

            row = {
                "imputer": imp_name,
                "model": model_name,
                "threshold": threshold,
                "constraint_satisfied": threshold_result["constraint_satisfied"],
                "best_params": best_params,
            }
            for metric_name, values in fold_metric_values.items():
                row[f"{metric_name}_mean"] = np.mean(values)
                row[f"{metric_name}_std"] = np.std(values)
            rows.append(row)

            print(f"  [{imp_name} | {model_name}] "
                  f"MCC={row['MCC_mean']:.3f}+/-{row['MCC_std']:.3f}  "
                  f"G_mean={row['G_mean_mean']:.3f}+/-{row['G_mean_std']:.3f}  "
                  f"constraint_ok={row['constraint_satisfied']}")

    comparison_df = pd.DataFrame(rows).set_index(["imputer", "model"])
    return comparison_df, oof_proba_dict




# =========================================================================== #
# TARGET LABEL ENCODING (handles string labels like "BBB+"/"BBB-")
# =========================================================================== #

def sanitize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Renames columns to be safe for every downstream model -- specifically
    LightGBM, which raises a hard error ("Do not support special JSON
    characters in feature name") on names containing characters like
    ( ) [ ] { } " : , since it stores feature names as part of its internal
    JSON model format. Common RDKit descriptor names (e.g. "fr_C(=O)O",
    which contains parentheses and an equals sign) trigger this.

    Any character that isn't a letter, digit, or underscore is replaced
    with "_"; if that collapses two different original names to the same
    sanitized name, a numeric suffix is appended to keep them unique.

    Returns (renamed_df, name_map) where name_map maps
    {sanitized_name: original_name}, so original descriptor names can still
    be reported/interpreted later even though the model was trained on the
    sanitized versions.
    """
    import re

    name_map = {}
    new_columns = []
    seen = {}
    for col in df.columns:
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
        if sanitized == "" or sanitized[0].isdigit():
            sanitized = f"f_{sanitized}"
        if sanitized in seen:
            seen[sanitized] += 1
            sanitized = f"{sanitized}_{seen[sanitized]}"
        else:
            seen[sanitized] = 0
        new_columns.append(sanitized)
        name_map[sanitized] = col

    renamed = df.copy()
    renamed.columns = new_columns
    return renamed, name_map


def encode_target_labels(y: pd.Series, positive_label=None) -> pd.Series:
    """
    Ensures y is numeric 0/1 for sklearn/dcor. If y is already numeric, it is
    returned unchanged (cast to int). If y holds string/categorical labels
    (e.g. "BBB+"/"BBB-", "Yes"/"No", "Permeable"/"Non-permeable"), it is
    mapped to 0/1.

    positive_label: which label to map to 1 (the positive class). If None,
    tries to infer it automatically:
      - exactly 2 unique labels and one contains "+" and the other "-"
        -> "+" label is positive
      - otherwise -> raises ValueError asking you to pass positive_label
        explicitly via config["positive_label"], since guessing wrong here
        silently flips TPR/TNR and every downstream metric.
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    uniques = sorted(y.dropna().unique().tolist())
    if len(uniques) != 2:
        raise ValueError(
            f"y_col has {len(uniques)} unique non-numeric labels {uniques}; "
            "expected exactly 2 for binary classification. Fix the column or "
            "pass config['positive_label'] explicitly."
        )

    if positive_label is None:
        plus = [u for u in uniques if "+" in str(u)]
        minus = [u for u in uniques if "-" in str(u)]
        if len(plus) == 1 and len(minus) == 1 and plus[0] != minus[0]:
            positive_label = plus[0]
        else:
            raise ValueError(
                f"Cannot infer which of {uniques} is the positive class. "
                "Pass config['positive_label'] explicitly (e.g. 'BBB+')."
            )

    if positive_label not in uniques:
        raise ValueError(f"positive_label={positive_label!r} not found in y_col values {uniques}")

    y_encoded = (y == positive_label).astype(int)
    print(f"Target encoding: {positive_label!r} -> 1, "
          f"{[u for u in uniques if u != positive_label][0]!r} -> 0")
    return y_encoded


def compute_missingness_report(
    X: pd.DataFrame, save_dir: str, filename: str = "train_descriptor_missingness.csv",
) -> pd.DataFrame:
    """
    Computes the percentage of missing values for every descriptor column in
    X (intended to be called on the TRAINING descriptor matrix only, right
    after the train/test split and before any downstream filtering,
    winsorization, or feature selection -- so the reported percentages
    reflect train-set missingness only, with no leakage from test).

    Saves a CSV to `save_dir/filename` with columns
    [descriptor, n_missing, n_total, missing_pct], sorted by missing_pct
    descending, and returns that same DataFrame.
    """
    import os

    n_total = len(X)
    n_missing = X.isna().sum()
    missing_pct = (n_missing / n_total * 100).round(3)

    report = pd.DataFrame({
        "descriptor": X.columns,
        "n_missing": n_missing.values,
        "n_total": n_total,
        "missing_pct": missing_pct.values,
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    report.to_csv(save_path, index=False)
    print(f"Missingness report ({len(report)} descriptors) saved to {save_path}")
    return report


def drop_high_missingness_descriptors(
    X_train: pd.DataFrame, X_test: pd.DataFrame, missingness_report: pd.DataFrame,
    threshold_pct: float = 70.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Drops descriptors whose TRAIN-set missingness exceeds `threshold_pct`
    (strictly greater than -- a descriptor at exactly the threshold is kept).
    The decision of WHICH columns to drop is made from train-set statistics
    only; the same columns are then dropped from X_test too (a fixed
    feature-set decision applied to both, not something re-derived from
    test -- no leakage).

    Returns (X_train_filtered, X_test_filtered, dropped_descriptor_names).
    """
    to_drop = missingness_report.loc[
        missingness_report["missing_pct"] > threshold_pct, "descriptor"
    ].tolist()
    to_drop = [c for c in to_drop if c in X_train.columns]

    X_train_filtered = X_train.drop(columns=to_drop)
    X_test_filtered = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])

    print(f"Missingness filter (> {threshold_pct}%): {len(X_train.columns)} -> "
          f"{len(X_train_filtered.columns)} descriptors ({len(to_drop)} dropped)")
    if to_drop:
        print(f"  Dropped for missingness > {threshold_pct}%: {to_drop}")

    return X_train_filtered, X_test_filtered, to_drop


def run_full_workflow(df: pd.DataFrame, config: dict) -> dict:
    """
    config keys (all the "/" choices):
      mw_col, mw_mode ("range"|"min_only"), mw_low, mw_high
        (MW is used to filter rows in Step 1, AND is kept as a descriptor
        column afterward -- it is not auto-excluded)
      smiles_col, y_col
      split_method ("random"|"scaffold"), test_size
      missingness_report_dir (default None -- if set, saves a per-descriptor
        train-set missingness % report there via compute_missingness_report)
      missingness_threshold_pct (default None -- if set, e.g. 70.0, drops
        descriptors whose TRAIN missingness exceeds this before winsorization,
        via drop_high_missingness_descriptors)
      dist_corr_threshold (default 0.25), spearman_threshold (default 0.85)
      winsorize (True|False)
      imputers (list from mean/median/KNN/MICE/class_specific)
      cv, calibration_method (default "isotonic"; pass "sigmoid" for Platt scaling)

    Order of operations: MW filter -> split -> [optional missingness report
    + high-missingness descriptor removal, TRAIN-derived] -> winsorization
    (bounds fit on train, frozen onto test) -> feature selection (fit on
    winsorized train) -> CV imputation/model comparison on the training set.
    """
    print("=== Step 1: MW filtering ===")
    df = filter_by_molecular_weight(
        df, mw_col=config.get("mw_col", "MW"), mode=config.get("mw_mode", "range"),
        low=config.get("mw_low", 150), high=config.get("mw_high", 800),
    )

    y_col = config["y_col"]
    smiles_col = config.get("smiles_col", "Smiles")
    # MW is kept as a descriptor here (used for row filtering in Step 1
    # above, but not auto-excluded from the descriptor set afterward).
    descriptor_cols = [c for c in df.columns if c not in {y_col, smiles_col}]

    # Sanitizing column names and encoding string labels to 0/1 are both
    # stateless, row-wise operations (no statistics computed across rows),
    # so doing them before the split is not a leakage concern.
    X_full = df[descriptor_cols]
    X_full, descriptor_name_map = sanitize_column_names(X_full)
    y = encode_target_labels(df[y_col], positive_label=config.get("positive_label"))

    print("\n=== Step 2: Train/test split ===")
    train_idx, test_idx = split_data(
        df, method=config.get("split_method", "scaffold"), smiles_col=smiles_col,
        y_col=y_col, test_size=config.get("test_size", 0.15),
    )
    X_train_full, X_test_full = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"train: {len(train_idx)}, test: {len(test_idx)}")
    train_balance = y_train.value_counts(normalize=True).round(3).to_dict()
    test_balance = y_test.value_counts(normalize=True).round(3).to_dict()
    print(f"train class proportions: {train_balance}")
    print(f"test class proportions:  {test_balance}")

    missingness_report = None
    dropped_for_missingness = []
    if config.get("missingness_report_dir"):
        print("\n=== Step 2b: Descriptor missingness in TRAIN set ===")
        missingness_report = compute_missingness_report(
            X_train_full, save_dir=config["missingness_report_dir"],
            filename=config.get("missingness_report_filename", "train_descriptor_missingness.csv"),
        )
        if config.get("missingness_threshold_pct") is not None:
            X_train_full, X_test_full, dropped_for_missingness = drop_high_missingness_descriptors(
                X_train_full, X_test_full, missingness_report,
                threshold_pct=config["missingness_threshold_pct"],
            )

    print("\n=== Step 3: Winsorization (bounds fit on TRAIN only) ===")
    X_train_full_before_winsor = X_train_full.copy()
    X_train_full, winsor_bounds = winsorize_columns(X_train_full, apply=config.get("winsorize", True))
    X_test_full = apply_frozen_winsor_bounds(X_test_full, winsor_bounds)  # frozen train bounds, not re-fit on test


    print("\n=== Step 4: Feature selection (fit on TRAIN only, post-winsorization) ===")
    dist_corr = select_features_distance_correlation(
        X_train_full, y_train, threshold=config.get("dist_corr_threshold", 0.25)
    )
    retained = remove_redundant_features_spearman(
        X_train_full, dist_corr, threshold=config.get("spearman_threshold", 0.85)
    )
    X_train = X_train_full[retained]
    X_test = X_test_full[retained]  # same columns applied to test, no re-fitting

    print("\n=== Steps 5-8: Imputation x Model comparison via single-level CV (impute+grid-search+calibrate+threshold+evaluate on the same cv folds) ===")
    imputer_names = config.get("imputers", ["mean", "median", "KNN", "MICE", "class_specific"])
    model_names = config.get(
        "models", ["RandomForest", "SVC", "LogisticRegression", "XGBoost", "CatBoost", "LightGBM"]
    )
    comparison, oof_proba_dict = run_cv_imputation_model_comparison(
        X_train, y_train, imputer_names, model_names,
        cv=config.get("cv", 5),
        calibration_method=config.get("calibration_method", "isotonic"),
        n_jobs=config.get("n_jobs", -1),
    )
    print(comparison.drop(columns=["best_params"]))



    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "retained_descriptors": retained,  # sanitized names -- what the models were actually trained on
        "retained_descriptors_original": [descriptor_name_map[c] for c in retained],  # human-readable
        "descriptor_name_map": descriptor_name_map,  # {sanitized: original}, covers ALL descriptors, not just retained
        "missingness_report": missingness_report,  # per-descriptor TRAIN missing % (None if not requested)
        "dropped_for_missingness": dropped_for_missingness,  # sanitized names dropped for missingness
        "winsor_bounds": winsor_bounds,
        "cv_comparison": comparison, "oof_proba_dict": oof_proba_dict,
        "calibration_fig":None,
    }


# =========================================================================== #
# Example / smoke test
# =========================================================================== #

def _example():
    rng = np.random.RandomState(RANDOM_STATE)
    n = 400
    smiles_pool = ["c1ccccc1O", "c1ccccc1C", "CCO", "CCN", "c1ccncc1", "C1CCCCC1", "CC(=O)O", "c1ccc2ccccc2c1"]

    df = pd.DataFrame({
        "Smiles": rng.choice(smiles_pool, n),
        "MW": rng.uniform(100, 900, n),
        "desc_1": rng.normal(size=n),
        "desc_2": rng.normal(size=n),
        "desc_3": rng.normal(loc=5, scale=2, size=n),
        "desc_4": rng.normal(size=n),  # noise, should get filtered by distance correlation
    })
    y = pd.Series((df["desc_1"] * 0.8 + df["desc_3"] * 0.3 + rng.normal(scale=1, size=n) > 3).astype(int))
    df["BBB"] = y

    for col in ["desc_1", "desc_2", "desc_3", "desc_4"]:
        mask = rng.rand(n) < 0.12
        df.loc[mask, col] = np.nan

    config = {
        "mw_col": "MW", "mw_mode": "range", "mw_low": 150, "mw_high": 800,
        "smiles_col": "Smiles", "y_col": "BBB",
        "dist_corr_threshold": 0.1,  # lowered for this small synthetic test
        "spearman_threshold": 0.85,
        "winsorize": True,
        "split_method": "scaffold", "test_size": 0.15,
        "imputers": ["mean", "median", "KNN", "class_specific"],  # MICE omitted for speed in smoke test
        "cv": 5, "calibration_method": "isotonic",
    }

    results = run_full_workflow(df, config)
    print("\n=== Retained descriptors ===")
    print(results["retained_descriptors"])


if __name__ == "__main__":
    _example()