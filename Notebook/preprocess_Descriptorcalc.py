"""
preprocess_Descriptorcalc.py

3D-structure preparation and descriptor-cleanup layer, sitting between raw
SMILES and Mordred descriptor calculation.

Full fallback ladder for 3D conformer generation, adapted from Demuth,
Schnizer & Svatunek's strainedSMILES2xyz workflow (J. Cheminform. 2026,
18:80, Fig. 2) -- ETKDGv3 ONLY throughout (the paper's ETKDGv2 fallback
tiers are intentionally NOT used here; every stage below uses ETKDGv3).

Stages, in order:
  1. ETKDGv3, default/deterministic (useRandomCoords=False), chirality
     enforced.
  --- STRAIN GATE: stages 2-5 only run if the molecule has a small
      strained ring (<12 atoms) containing a double/triple bond, per
      _structural_flags()'s has_small_strained_ring -- the paper's
      stated condition for engaging the expensive fallback ladder at
      all. Molecules failing stage 1 for unrelated reasons stop here
      and are diagnosed directly, not retried through the whole ladder. ---
  2. ETKDGv3 + random coordinates, chirality still enforced.
  3. ETKDGv3 + random coords + relaxed constraints (basic knowledge
     off, smoothing failures ignored), chirality still enforced.
  4. Systematic double-bond stereo inversion + retry: for each stereo
     double bond in the molecule, flip its E/Z assignment one at a
     time and retry stage 3's parameters. Accepts the FIRST successful
     flip. Flagged distinctly (status "ok_stereo_inverted:<bond_idx>")
     since the resulting geometry corresponds to a DIFFERENT
     stereoisomer than the input SMILES.
  5. ETKDGv3 + random + relaxed + NO chirality enforcement (final
     stage). Flagged "ok_chirality_unenforced" -- EXCLUDED from
     df_3d_ready below: since ORCA is never initiated on these, they
     are excluded from the study entirely, not just from descriptor
     calculation.

Every compound that fails ALL applicable stages gets an automatic,
per-compound diagnostic (RDKit's trackFailures mechanism, same approach
as diagnose_embedding_failures() below) printed immediately -- not a
separate manual re-diagnosis step.

NOTE ON RDKIT VERSION COMPATIBILITY:
RDKit's params.trackFailures / params.GetFailureCounts() /
Chem.rdDistGeom.EmbedFailureCauses diagnostic mechanism is only present
in newer RDKit builds. Every use of it below is guarded with hasattr()
checks so this file runs cleanly on older RDKit installs too -- on those,
you just lose the structured "why did embedding fail" breakdown and get
a generic "no_specific_cause_identified" instead.
"""

import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def _capture_rdkit_stderr(fn, *args, **kwargs):
    STDERR_FD = 2
    saved_fd = os.dup(STDERR_FD)
    r, w = os.pipe()
    os.dup2(w, STDERR_FD)
    os.close(r)

    try:
        result = fn(*args, **kwargs)
    finally:
        os.dup2(saved_fd, STDERR_FD)
        os.close(saved_fd)
        captured = os.read(r, 200_000).decode(errors="ignore")
        os.close(r)

    return result, captured.strip()


def _structural_flags(mol: Chem.Mol) -> dict:
    """Structural properties commonly associated with embedding failure --
    useful even when RDKit emits no explicit message."""
    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]

    has_small_strained_ring = False
    for ring in ri.AtomRings():
        if len(ring) < 12:
            for bond in mol.GetBonds():
                if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
                    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if a1 in ring and a2 in ring:
                        has_small_strained_ring = True
                        break

    return {
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_rings": ri.NumRings(),
        "max_ring_size": max(ring_sizes) if ring_sizes else 0,
        "has_macrocycle_12plus": any(s >= 12 for s in ring_sizes),
        "has_small_strained_ring": has_small_strained_ring,  # per strainedSMILES2xyz's definition
        "n_stereocenters": len(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)),
    }


def embed_only(smiles: str, seed: int = 42):
    """
    Runs the full ETKDGv3-only fallback ladder (stages 1-5 above).
    Returns (mol_or_None, status), where status is one of:
      "invalid_smiles"
      "ok"                       -- stage 1 succeeded (strict, deterministic)
      "ok_random_coords"         -- stage 2 succeeded
      "ok_relaxed"               -- stage 3 succeeded
      "ok_stereo_inverted:<idx>" -- stage 4 succeeded (bond idx flipped --
                                     review stereochemistry before trusting)
      "ok_chirality_unenforced"  -- stage 5 succeeded (chirality NOT
                                     enforced -- EXCLUDED downstream)
      "not_strained_no_ladder"   -- stage 1 failed AND the strain gate
                                     didn't engage stages 2-5 (molecule
                                     doesn't have a small strained ring)
      "embedding_failed:<cause> (breakdown: {...})" -- every applicable
                                     stage failed; <cause> is RDKit's
                                     dominant internal failure reason
                                     when available (via trackFailures,
                                     same mechanism as
                                     diagnose_embedding_failures()), or
                                     "no_specific_cause_identified" on
                                     older RDKit builds without that
                                     diagnostic mechanism.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    # --- Stage 1: ETKDGv3, strict/deterministic, chirality enforced ---
    mol_h1 = Chem.AddHs(mol)
    p1 = AllChem.ETKDGv3()
    p1.randomSeed = seed
    p1.useRandomCoords = False
    p1.enforceChirality = True
    if AllChem.EmbedMolecule(mol_h1, p1) == 0 and mol_h1.GetNumConformers() > 0:
        return mol_h1, "ok"

    # --- Strain gate: only escalate the ladder for small strained rings
    # with a double/triple bond, per strainedSMILES2xyz's definition ---
    if not _structural_flags(mol)["has_small_strained_ring"]:
        return None, "not_strained_no_ladder"

    # --- Stage 2: + random coords, chirality still enforced ---
    mol_h2 = Chem.AddHs(mol)
    p2 = AllChem.ETKDGv3()
    p2.randomSeed = seed
    p2.useRandomCoords = True
    p2.enforceChirality = True
    if AllChem.EmbedMolecule(mol_h2, p2) == 0 and mol_h2.GetNumConformers() > 0:
        return mol_h2, "ok_random_coords"

    # --- Stage 3: + relaxed constraints, chirality still enforced ---
    mol_h3 = Chem.AddHs(mol)
    p3 = AllChem.ETKDGv3()
    p3.randomSeed = seed
    p3.useRandomCoords = True
    p3.useBasicKnowledge = False
    p3.ignoreSmoothingFailures = True
    p3.enforceChirality = True
    if AllChem.EmbedMolecule(mol_h3, p3) == 0 and mol_h3.GetNumConformers() > 0:
        return mol_h3, "ok_relaxed"

    # --- Stage 4: systematic double-bond stereo inversion + retry ---
    stereo_bond_idxs = [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() != Chem.BondStereo.STEREONONE
    ]
    flip_map = {
        Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS,
    }
    for bond_idx in stereo_bond_idxs:
        mol_flipped = Chem.Mol(mol)
        bond = mol_flipped.GetBondWithIdx(bond_idx)
        current = bond.GetStereo()
        if current not in flip_map:
            continue
        bond.SetStereo(flip_map[current])

        mol_h4 = Chem.AddHs(mol_flipped)
        p4 = AllChem.ETKDGv3()
        p4.randomSeed = seed
        p4.useRandomCoords = True
        p4.useBasicKnowledge = False
        p4.ignoreSmoothingFailures = True
        p4.enforceChirality = True
        if AllChem.EmbedMolecule(mol_h4, p4) == 0 and mol_h4.GetNumConformers() > 0:
            return mol_h4, f"ok_stereo_inverted:{bond_idx}"

    # --- Stage 5: + NO chirality enforcement (final stage) ---
    mol_h5 = Chem.AddHs(mol)
    p5 = AllChem.ETKDGv3()
    p5.randomSeed = seed
    p5.useRandomCoords = True
    p5.useBasicKnowledge = False
    p5.ignoreSmoothingFailures = True
    p5.enforceChirality = False
    if AllChem.EmbedMolecule(mol_h5, p5) == 0 and mol_h5.GetNumConformers() > 0:
        return mol_h5, "ok_chirality_unenforced"  # EXCLUDED downstream -- see prepare_3D_dataframe_twopass

    # --- All applicable stages exhausted: diagnose and report per-compound.
    # Guarded for RDKit builds that don't have trackFailures / GetFailureCounts
    # / EmbedFailureCauses at all (this feature was added in a later RDKit
    # release than some environments have installed). ---
    mol_diag = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p_diag = AllChem.ETKDGv3()
    p_diag.randomSeed = seed
    if hasattr(p_diag, "trackFailures"):
        p_diag.trackFailures = True
    AllChem.EmbedMolecule(mol_diag, p_diag)

    if hasattr(Chem.rdDistGeom, "EmbedFailureCauses") and hasattr(p_diag, "GetFailureCounts"):
        failure_cause_names = list(Chem.rdDistGeom.EmbedFailureCauses.names.keys())
        failure_counts = p_diag.GetFailureCounts()
        causes = {name: c for name, c in zip(failure_cause_names, failure_counts) if c > 0}
    else:
        causes = {}

    dominant = max(causes, key=causes.get) if causes else "no_specific_cause_identified"
    diagnosis = f"embedding_failed:{dominant} (breakdown: {causes})"
    print(f"  [FAILED] {smiles}: {diagnosis}")
    return None, diagnosis


def minimize_only(mol, max_iters: int = 200):
    """
    Pass 2: force-field optimization on an ALREADY-EMBEDDED molecule.
    Checks parameter availability before calling either force field, to
    avoid the 'bad params pointer' crash from calling UFF on unsupported
    atom types.
    Returns (mol_or_None, status).
    """
    if mol is None:
        return None, "no_input_mol"

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=max_iters)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        else:
            return None, "no_force_field_params"
    except RuntimeError as e:
        return None, f"optimization_failed: {e}"

    return mol, "ok"


# Statuses that still ENFORCE chirality, and so are usable for descriptor
# calculation / passed on to ORCA. "ok_chirality_unenforced" is
# DELIBERATELY excluded -- ORCA is never initiated on these, so they are
# excluded from the study entirely, not just from this dataframe.
_CHIRALITY_ENFORCED_STATUSES = {"ok", "ok_random_coords", "ok_relaxed"}


def prepare_3D_dataframe_twopass(df, smiles_col: str = "SMILES"):
    """
    Combines both passes on a dataframe, with SEPARATE attrition tables for
    embedding vs. minimization -- report both in methods.

    Returns (df_3d_ready, embed_attrition, minimize_attrition, df_full)
      - df_3d_ready: only compounds that enforced chirality at every
        applicable stage AND minimized successfully -- ready for
        descriptor calculation / ORCA.
      - df_full: EVERY input row, with embed_status/minimize_status
        columns intact -- use this to retrieve the specific failed rows,
        e.g. df_full[df_full["embed_status"] != "ok"].
    """
    df = df.copy()

    # --- Pass 1: embed everything, full ladder ---
    embed_results = df[smiles_col].apply(embed_only)
    df["mol_embedded"], df["embed_status"] = zip(*embed_results)

    embed_attrition = df["embed_status"].value_counts()
    print("=== Embedding attrition (Pass 1, full ETKDGv3-only ladder) ===")
    print(embed_attrition)

    n_chirality_unenforced = (df["embed_status"] == "ok_chirality_unenforced").sum()
    n_stereo_inverted = df["embed_status"].str.startswith("ok_stereo_inverted", na=False).sum()
    print(f"\n{n_chirality_unenforced} compound(s) succeeded ONLY via the final stage "
          f"(chirality unenforced) -- EXCLUDED from the study; ORCA will not be initiated on these.")
    print(f"{n_stereo_inverted} compound(s) succeeded via double-bond stereo inversion "
          f"-- KEPT, but flag for stereochemistry review before ORCA.")

    is_usable = df["embed_status"].isin(_CHIRALITY_ENFORCED_STATUSES) | \
        df["embed_status"].str.startswith("ok_stereo_inverted", na=False)
    embedded = df[is_usable].copy()

    # --- Pass 2: minimize only the usable Pass-1 successes ---
    minimize_results = embedded["mol_embedded"].apply(minimize_only)
    embedded["mol_final"], embedded["minimize_status"] = zip(*minimize_results)

    minimize_attrition = embedded["minimize_status"].value_counts()
    print("\n=== Minimization attrition (Pass 2, run only on usable Pass-1 successes) ===")
    print(minimize_attrition)

    # fold minimize_status back into the FULL dataframe (Pass-1 failures
    # AND chirality-unenforced successes get "not_attempted")
    df["minimize_status"] = "not_attempted"
    df.loc[embedded.index, "minimize_status"] = embedded["minimize_status"]

    df_3d_ready = embedded[embedded["minimize_status"] == "ok"].copy()
    df_3d_ready["mol_3d"] = df_3d_ready["mol_final"]
    df_3d_ready = df_3d_ready.drop(columns=["mol_embedded", "mol_final"])

    print(f"\nFinal: {len(df_3d_ready)} / {len(df)} compounds ready for descriptor calculation")

    return df_3d_ready, embed_attrition, minimize_attrition, df


def diagnose_embedding_failures(smiles_list, seed: int = 42):
    """
    Re-runs embedding on a list of previously-failed SMILES, using RDKit's
    BUILT-IN failure tracking (params.trackFailures=True +
    params.GetFailureCounts()) when available, rather than capturing
    printed stderr text.

    This is a real, structured breakdown of WHY embedding failed --
    an enumerated count of which internal check failed and how often --
    when your installed RDKit build supports it. On older RDKit builds
    that lack trackFailures/GetFailureCounts/EmbedFailureCauses entirely,
    this falls back to reporting only whether embedding succeeded, with
    no structured cause breakdown (dominant_failure_cause / breakdown
    both come back None).

    Common failure causes (from rdkit.Chem.rdDistGeom.EmbedFailureCauses,
    when available):
      INITIAL_COORDS         -- couldn't generate a starting coordinate set
      FIRST_MINIMIZATION     -- failed the first geometry minimization pass
      CHECK_TETRAHEDRAL_CENTERS / CHECK_CHIRAL_CENTERS / CHECK_CHIRAL_CENTERS2
                             -- generated geometry violates specified stereochemistry
      FINAL_CHIRAL_BOUNDS    -- final geometry fails the chirality bounds check
      BAD_DOUBLE_BOND_STEREO / LINEAR_DOUBLE_BOND -- double bond geometry issues
      MINIMIZATION / ETK_MINIMIZATION -- force-field minimization didn't converge
      CLASH                  -- unresolvable atomic clashes
      EXCEEDED_TIMEOUT       -- ran out of time (only relevant if params.timeout is set)

    Run this ONLY on your failed compounds, not the full dataset.
    """
    has_tracking = hasattr(Chem.rdDistGeom, "EmbedFailureCauses")
    if has_tracking:
        failure_cause_names = list(Chem.rdDistGeom.EmbedFailureCauses.names.keys())

    rows = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({"smiles": smi, "diagnosis": "invalid_smiles"})
            continue

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if has_tracking and hasattr(params, "trackFailures"):
            params.trackFailures = True

        result = AllChem.EmbedMolecule(mol_h, params)

        if has_tracking and hasattr(params, "GetFailureCounts"):
            failure_counts = params.GetFailureCounts()
            dominant_causes = {
                name: count for name, count in zip(failure_cause_names, failure_counts) if count > 0
            }
        else:
            dominant_causes = None

        rows.append({
            "smiles": smi,
            "embedding_succeeded": result == 0,
            "dominant_failure_cause": max(dominant_causes, key=dominant_causes.get) if dominant_causes else None,
            "failure_cause_breakdown": dominant_causes if dominant_causes else None,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Missing 3D descriptor detection (ignore_3D=False pathway)
# --------------------------------------------------------------------------- #

def get_3d_only_descriptor_names(calc_with_3d=None) -> set:
    """
    Returns the set of descriptor names that ONLY exist when ignore_3D=False
    (i.e. genuinely require a 3D conformer) -- computed by diffing against
    an ignore_3D=True calculator. Needed because Mordred's `Missing` type
    is also used for UNRELATED failures (e.g. small-molecule autocorrelation
    descriptors dividing by zero at high topological lags), which have
    nothing to do with conformer availability and would otherwise
    contaminate the diagnostic.
    """
    from mordred import Calculator, descriptors

    calc_2d = Calculator(descriptors, ignore_3D=True)
    calc_3d = calc_with_3d or Calculator(descriptors, ignore_3D=False)
    names_2d = set(str(d) for d in calc_2d.descriptors)
    names_3d = set(str(d) for d in calc_3d.descriptors)
    return names_3d - names_2d


def count_missing_3d_descriptors(df_mordred: pd.DataFrame):
    """
    For a dataframe produced by calc.pandas(mols) with ignore_3D=False,
    counts Mordred's `Missing` objects per row and per column, RESTRICTED
    to the columns that are genuinely 3D-dependent (per
    get_3d_only_descriptor_names()). This avoids false positives from
    unrelated 2D-descriptor computation failures (e.g. autocorrelation
    descriptors on very small molecules), which also use the same
    `Missing` sentinel type but have nothing to do with conformers.

    Returns (missing_per_row, missing_per_column), both restricted to
    genuine 3D descriptor columns.
    """
    import mordred.error

    plain_df = pd.DataFrame(df_mordred)  # MordredDataFrame lacks .applymap/.map directly
    threed_cols = [c for c in plain_df.columns if c in get_3d_only_descriptor_names()]
    threed_df = plain_df[threed_cols]

    is_missing = threed_df.applymap(lambda x: isinstance(x, mordred.error.Missing))

    missing_per_row = is_missing.sum(axis=1)
    missing_per_column = is_missing.sum(axis=0)
    missing_per_column = missing_per_column[missing_per_column > 0].sort_values(ascending=False)

    return missing_per_row, missing_per_column


def clean_mordred_missing(df_mordred: pd.DataFrame) -> pd.DataFrame:
    """Converts Mordred's Missing objects (and any other non-numeric error
    objects) to proper NaN, so the dataframe is safe for imputation/scaling."""
    return df_mordred.apply(pd.to_numeric, errors="coerce")