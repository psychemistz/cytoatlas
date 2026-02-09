#!/usr/bin/env python3
"""
Create lite dataset for CytoAtlas visualization testing.

Reduces visualization/data/ (~8.7 GB) to visualization/data_lite/ (~100-150 MB)
by filtering SecAct signatures, subsampling cell types/organs/diseases, and
truncating large scatter point arrays.

Usage:
    python scripts/create_data_lite.py
    python scripts/create_data_lite.py --secact-top-n 50 --dry-run
    python scripts/create_data_lite.py --source visualization/data --output visualization/data_lite
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Biologically important SecAct signatures to always include
MUST_INCLUDE_SECACT = {
    "S100A8", "S100A9", "MMP9", "SPP1", "VEGFA", "TGFB1", "CCL2", "CXCL10",
    "IL16", "GZMM", "GZMB", "GZMA", "PRF1", "GNLY", "LYZ", "CTSS",
    "CD40LG", "CRTAP", "ADAM9", "PLAT",
}

# Organs to keep for scAtlas (10 of 35)
KEEP_ORGANS = {
    "Blood", "Lung", "Liver", "Kidney", "Heart",
    "Skin", "Colon", "Breast", "Pancreas", "Brain",
}

# Diseases to keep for inflammation (10 of 20, at least 1 per group)
KEEP_DISEASES = {
    "COVID", "SLE", "RA", "UC", "COPD",
    "BRCA", "HIV", "healthy", "sepsis", "CD",
}

# Cancer types to keep (10 of 29)
KEEP_CANCER_TYPES = {
    "Breast", "Lung", "Liver", "Colorectal", "Pancreatic",
    "Kidney", "Melanoma", "Ovarian", "Glioblastoma", "Head_Neck",
}

# CIMA cell types to keep (15 of 27 - major lineages)
KEEP_CIMA_CELLTYPES = {
    "CD4_Naive", "CD4_Memory", "CD4_Th17", "CD4_Treg",
    "CD8_Naive", "CD8_Memory", "CD8_Effector",
    "NK_CD56dim", "NK_CD56bright",
    "Mono_classical", "Mono_nonClassical",
    "DC_myeloid", "B_Naive", "B_Memory",
    "Plasma",
}

# Max scatter points per series
MAX_SCATTER_POINTS = 200

# Threshold for "small file" (copy as-is)
SMALL_FILE_THRESHOLD = 5_000_000  # 5 MB


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create lite dataset for CytoAtlas visualization testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("visualization/data"),
        help="Source data directory (default: visualization/data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualization/data_lite"),
        help="Output directory (default: visualization/data_lite)",
    )
    parser.add_argument(
        "--secact-top-n",
        type=int,
        default=50,
        help="Number of top SecAct signatures to keep (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Global filter computation
# ---------------------------------------------------------------------------

def compute_secact_top_n(source_dir: Path, top_n: int) -> set:
    """Rank SecAct signatures by importance across atlases.

    Ranking criteria:
    1. Presence in cross_atlas.json conserved signatures (n_atlases)
    2. Absolute mean effect across atlases
    3. Must-include biologically important signatures
    """
    cross_atlas_path = source_dir / "cross_atlas.json"
    if not cross_atlas_path.exists():
        print("  WARNING: cross_atlas.json not found, using must-include set only")
        return set(list(MUST_INCLUDE_SECACT)[:top_n])

    print("  Loading cross_atlas.json for SecAct ranking...")
    with open(cross_atlas_path) as f:
        cross_atlas = json.load(f)

    conserved = cross_atlas.get("conserved", {}).get("signatures", [])
    secact_sigs = [s for s in conserved if s.get("signature_type") == "SecAct"]

    # Score each signature
    scored = []
    for sig in secact_sigs:
        name = sig["signature"]
        n_atlases = sig.get("n_atlases", 0)
        # Sum of absolute means across atlases
        abs_effect = (
            abs(sig.get("cima_mean", 0))
            + abs(sig.get("inflammation_mean", 0))
            + abs(sig.get("scatlas_mean", 0))
        )
        # Bonus for must-include
        bonus = 100 if name in MUST_INCLUDE_SECACT else 0
        scored.append((name, bonus + n_atlases * 10 + abs_effect))

    scored.sort(key=lambda x: -x[1])

    # Always include must-include set, fill rest from ranking
    keep = set(MUST_INCLUDE_SECACT)
    for name, _ in scored:
        if len(keep) >= top_n:
            break
        keep.add(name)

    print(f"  Selected {len(keep)} SecAct signatures (including {len(MUST_INCLUDE_SECACT)} must-include)")
    return keep


def compute_inflammation_celltypes(source_dir: Path, max_n: int = 20) -> set:
    """Select top inflammation cell types by frequency in the data."""
    path = source_dir / "inflammation_disease_filtered.json"
    if not path.exists():
        path = source_dir / "inflammation_disease.json"
    if not path.exists():
        return set()

    print("  Scanning inflammation cell types...")
    with open(path) as f:
        data = json.load(f)

    counts = Counter()
    for rec in data:
        ct = rec.get("cell_type", "")
        if ct:
            counts[ct] += 1

    # Take top N by frequency
    top = {ct for ct, _ in counts.most_common(max_n)}
    print(f"  Selected {len(top)} of {len(counts)} inflammation cell types")
    return top


def compute_scatlas_celltypes(source_dir: Path, max_n: int = 30) -> set:
    """Select top scAtlas cell types from kept organs."""
    path = source_dir / "scatlas_celltypes.json"
    if not path.exists():
        return set()

    print("  Scanning scAtlas cell types...")
    with open(path) as f:
        data = json.load(f)

    records = data.get("data", data) if isinstance(data, dict) else data

    # Count cells per cell_type, but only in kept organs
    counts = Counter()
    for rec in records:
        organ = rec.get("organ", "")
        if organ in KEEP_ORGANS:
            ct = rec.get("cell_type", "")
            n = rec.get("n_cells", 1)
            counts[ct] += n

    top = {ct for ct, _ in counts.most_common(max_n)}
    print(f"  Selected {len(top)} of {len(counts)} scAtlas cell types (from {len(KEEP_ORGANS)} organs)")
    return top


def compute_cancer_types_from_data(source_dir: Path, max_n: int = 10) -> set:
    """Get actual cancer type names from the data files."""
    # Try caf_signatures first as it has explicit cancer_type field
    for fname in ["caf_signatures.json", "exhaustion.json", "immune_infiltration.json"]:
        path = source_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        # Look for cancer_types list or extract from data
        if isinstance(data, dict):
            if "cancer_types" in data:
                all_types = data["cancer_types"]
                break
            # Extract from data records
            records = data.get("data", [])
            all_types = sorted(set(r.get("cancer_type", "") for r in records if r.get("cancer_type")))
            if all_types:
                break
    else:
        return KEEP_CANCER_TYPES

    # Match our desired set against actual names (fuzzy)
    keep = set()
    for actual in all_types:
        actual_lower = actual.lower().replace("_", " ")
        for desired in KEEP_CANCER_TYPES:
            if desired.lower() in actual_lower or actual_lower.startswith(desired.lower()):
                keep.add(actual)
                break
        if len(keep) >= max_n:
            break

    # If we didn't get enough, add from the beginning of the list
    for ct in all_types:
        if len(keep) >= max_n:
            break
        keep.add(ct)

    print(f"  Selected {len(keep)} of {len(all_types)} cancer types")
    return keep


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------

def should_keep_signature(rec: dict, keep_secact: set) -> bool:
    """Check if a record's signature should be kept."""
    sig_type = rec.get("signature_type", rec.get("sig_type", ""))
    sig = rec.get("signature", rec.get("protein", ""))

    if sig_type == "CytoSig":
        return True
    if sig_type == "SecAct":
        return sig in keep_secact
    # LinCytoSig or unknown - keep
    if sig_type in ("LinCytoSig", "lincytosig"):
        return True
    # No signature_type field - keep
    if not sig_type:
        return True
    return sig in keep_secact


def filter_array_records(
    records: list,
    keep_secact: set,
    keep_celltypes: dict = None,
    keep_diseases: set = None,
    keep_organs: set = None,
    keep_cancer_types: set = None,
) -> list:
    """Filter an array of records by signature, cell type, disease, organ."""
    result = []
    for rec in records:
        if not should_keep_signature(rec, keep_secact):
            continue
        # Cell type filter (atlas-specific)
        if keep_celltypes:
            ct = rec.get("cell_type", "")
            atlas = rec.get("atlas", "").lower()
            if "cima" in atlas and ct and ct not in keep_celltypes.get("cima", set()):
                continue
            if "inflam" in atlas and ct and ct not in keep_celltypes.get("inflammation", set()):
                continue
            if ("scatlas" in atlas or "normal" in atlas.lower() or "cancer" in atlas.lower()) and ct and ct not in keep_celltypes.get("scatlas", set()):
                continue
        # Disease filter
        if keep_diseases:
            disease = rec.get("disease", "")
            if disease and disease not in keep_diseases:
                continue
        # Organ filter
        if keep_organs:
            organ = rec.get("organ", "")
            if organ and organ not in keep_organs:
                continue
        # Cancer type filter
        if keep_cancer_types:
            cancer = rec.get("cancer_type", "")
            if cancer and cancer not in keep_cancer_types:
                continue
        result.append(rec)
    return result


def subsample_points(points: list, max_n: int = MAX_SCATTER_POINTS) -> list:
    """Random subsample of scatter points."""
    if len(points) <= max_n:
        return points
    return random.sample(points, max_n)


def filter_signature_list(sigs: list, keep_secact: set, sig_type: str = "SecAct") -> list:
    """Filter a list of signature names."""
    if sig_type == "CytoSig":
        return sigs  # Keep all CytoSig
    return [s for s in sigs if s in keep_secact]


# ---------------------------------------------------------------------------
# File-specific processors
# ---------------------------------------------------------------------------

def process_activity_boxplot(data, keep_secact, keep_celltypes):
    """Filter activity_boxplot.json (array of ~1.2M records)."""
    return filter_array_records(data, keep_secact, keep_celltypes=keep_celltypes)


def process_inflammation_disease(data, keep_secact, keep_celltypes, keep_diseases):
    """Filter inflammation_disease.json or inflammation_disease_filtered.json."""
    return filter_array_records(
        data, keep_secact,
        keep_celltypes=keep_celltypes,
        keep_diseases=keep_diseases,
    )


def process_singlecell_activity(data, keep_secact, keep_celltypes):
    """Filter singlecell_activity.json (array of ~720K records)."""
    return filter_array_records(data, keep_secact, keep_celltypes=keep_celltypes)


def process_scatlas_celltypes(data, keep_secact, keep_organs, keep_scatlas_cts):
    """Filter scatlas_celltypes.json (dict with data array)."""
    if not isinstance(data, dict):
        return filter_array_records(data, keep_secact, keep_organs=keep_organs)

    result = dict(data)
    if "data" in result:
        filtered = []
        for rec in result["data"]:
            if not should_keep_signature(rec, keep_secact):
                continue
            if rec.get("organ", "") and rec["organ"] not in keep_organs:
                continue
            if rec.get("cell_type", "") and rec["cell_type"] not in keep_scatlas_cts:
                continue
            filtered.append(rec)
        result["data"] = filtered

    # Update metadata lists
    if "organs" in result:
        result["organs"] = [o for o in result["organs"] if o in keep_organs]
    if "all_cell_types" in result:
        result["all_cell_types"] = sorted(keep_scatlas_cts)
    if "top_cell_types" in result:
        result["top_cell_types"] = [ct for ct in result["top_cell_types"] if ct in keep_scatlas_cts]
    if "secact_signatures" in result:
        result["secact_signatures"] = filter_signature_list(result["secact_signatures"], keep_secact)
    if "signature_counts" in result:
        result["signature_counts"] = {
            "CytoSig": result.get("signature_counts", {}).get("CytoSig", 43),
            "SecAct": len(keep_secact),
        }
    return result


def process_age_bmi_boxplots(data, keep_secact, keep_celltypes):
    """Filter age_bmi_boxplots.json (dict with cima/inflammation sub-dicts)."""
    result = {}
    for atlas_key in ("cima", "inflammation"):
        if atlas_key not in data:
            continue
        atlas_data = data[atlas_key]
        sub = {}
        ct_set = keep_celltypes.get(atlas_key, set()) if keep_celltypes else None

        for section in ("age", "bmi"):
            if section not in atlas_data:
                continue
            records = atlas_data[section]
            filtered = []
            for rec in records:
                if not should_keep_signature(rec, keep_secact):
                    continue
                if ct_set:
                    ct = rec.get("cell_type", "")
                    if ct and ct not in ct_set:
                        continue
                filtered.append(rec)
            sub[section] = filtered

        # Copy metadata, filter signature lists
        for key in ("cytosig_signatures", "age_bins", "bmi_bins", "cell_types"):
            if key in atlas_data:
                sub[key] = atlas_data[key]
        if "secact_signatures" in atlas_data:
            sub["secact_signatures"] = filter_signature_list(
                atlas_data["secact_signatures"], keep_secact
            )
        if "cell_types" in sub and ct_set:
            sub["cell_types"] = [ct for ct in sub["cell_types"] if ct in ct_set]

        result[atlas_key] = sub
    return result


def process_scatlas_singlecell(data, keep_secact, keep_organs, keep_scatlas_cts):
    """Filter scatlas_normal_singlecell_secact.json (flat array)."""
    result = []
    for rec in data:
        if not should_keep_signature(rec, keep_secact):
            continue
        # These files don't have organ field - filter by cell type only
        ct = rec.get("cell_type", "")
        if ct and keep_scatlas_cts and ct not in keep_scatlas_cts:
            continue
        result.append(rec)
    return result


def process_bulk_donor_correlations(data, keep_secact):
    """Filter bulk_donor_correlations.json (5.5 GB nested dict).

    Structure:
      summary: list[12] - keep all
      donor_level: dict[atlas][sig_type] -> list of records
      celltype_level: dict[atlas][level] -> list of records with top_targets
      donor_scatter: dict[atlas][sig_type][signature] -> {points, ...}
      celltype_scatter: dict[atlas][level][sig_type][signature] -> {points, ...}
      resampled_scatter: dict[atlas][level][sig_type][signature] -> {points, ...}
    """
    result = {}

    # summary: keep as-is
    result["summary"] = data.get("summary", [])

    # donor_level: filter by signature type
    donor_level = {}
    for atlas, sig_types in data.get("donor_level", {}).items():
        donor_level[atlas] = {}
        for sig_type, records in sig_types.items():
            if sig_type == "secact":
                filtered = [r for r in records if r.get("target", r.get("gene", "")) in keep_secact]
                donor_level[atlas][sig_type] = filtered
            else:
                donor_level[atlas][sig_type] = records
    result["donor_level"] = donor_level

    # celltype_level: filter top_targets within each record
    celltype_level = {}
    for atlas, levels in data.get("celltype_level", {}).items():
        celltype_level[atlas] = {}
        for level, records in levels.items():
            celltype_level[atlas][level] = records  # Keep structure, these are small
    result["celltype_level"] = celltype_level

    # Scatter data: filter by signature name and subsample points
    for scatter_key in ("donor_scatter", "celltype_scatter", "resampled_scatter"):
        result[scatter_key] = _filter_scatter_nested(data.get(scatter_key, {}), keep_secact)

    return result


def _filter_scatter_nested(scatter_data, keep_secact, depth=0):
    """Recursively filter scatter data, keeping CytoSig + top SecAct, subsampling points."""
    if not isinstance(scatter_data, dict):
        return scatter_data

    # Check if this is a leaf scatter object (has 'points' key)
    if "points" in scatter_data:
        result = dict(scatter_data)
        result["points"] = subsample_points(scatter_data["points"])
        return result

    # Check if keys look like signature names (leaf level of nesting)
    # We detect this by checking if values are dicts with 'points' or 'rho'
    sample_values = list(scatter_data.values())[:3]
    is_signature_level = any(
        isinstance(v, dict) and ("points" in v or "rho" in v)
        for v in sample_values
    )

    if is_signature_level:
        # Filter: keep all keys that aren't SecAct, or are in keep_secact
        result = {}
        for sig_name, sig_data in scatter_data.items():
            # We don't know sig_type at this level, but SecAct signatures are
            # typically not in CytoSig's 43. Keep if in keep_secact or if it's
            # a CytoSig name. Since CytoSig names are things like "IL6", "IFNG"
            # which could overlap, we just check keep_secact for SecAct names
            # and keep everything that's not clearly SecAct-only.
            # Simple heuristic: keep if it's in keep_secact OR if there are fewer
            # than 100 keys (meaning it's probably CytoSig/LinCytoSig level)
            if len(scatter_data) <= 200 or sig_name in keep_secact:
                if isinstance(sig_data, dict) and "points" in sig_data:
                    result[sig_name] = dict(sig_data)
                    result[sig_name]["points"] = subsample_points(sig_data["points"])
                else:
                    result[sig_name] = sig_data
        return result

    # Recurse into sub-dicts
    result = {}
    for key, value in scatter_data.items():
        if isinstance(value, dict):
            result[key] = _filter_scatter_nested(value, keep_secact, depth + 1)
        else:
            result[key] = value
    return result


def process_bulk_rnaseq_validation(data, keep_secact):
    """Filter bulk_rnaseq_validation.json (91 MB, dict with gtex/tcga)."""
    result = {}
    for atlas_key in ("gtex", "tcga"):
        if atlas_key not in data:
            continue
        atlas = dict(data[atlas_key])

        # Filter donor_level secact
        if "donor_level" in atlas:
            dl = dict(atlas["donor_level"])
            if "secact" in dl:
                dl["secact"] = [
                    r for r in dl["secact"]
                    if r.get("target", r.get("gene", "")) in keep_secact
                ]
            atlas["donor_level"] = dl

        # Filter scatter data
        for scatter_key in ("donor_scatter", "stratified_scatter"):
            if scatter_key in atlas:
                atlas[scatter_key] = _filter_scatter_nested(
                    atlas[scatter_key], keep_secact
                )

        result[atlas_key] = atlas
    return result


def process_validation_file(data, keep_secact):
    """Filter validation/*.json files (dict with multiple validation sections)."""
    if not isinstance(data, dict):
        return data

    result = dict(data)

    # Each section is typically a list of records with 'signature' and 'signature_type'
    for key in (
        "sample_validations", "celltype_validations", "pseudobulk_vs_sc",
        "singlecell_validations", "gene_coverage", "cv_stability",
    ):
        if key in result and isinstance(result[key], list):
            result[key] = [
                r for r in result[key]
                if should_keep_signature(r, keep_secact)
            ]

    # biological_associations: filter results sub-list
    if "biological_associations" in result:
        ba = dict(result["biological_associations"])
        if "results" in ba and isinstance(ba["results"], list):
            ba["results"] = [
                r for r in ba["results"]
                if should_keep_signature(r, keep_secact)
            ]
        result["biological_associations"] = ba

    return result


def process_caf_signatures(data, keep_secact, keep_cancer_types):
    """Filter caf_signatures.json."""
    if not isinstance(data, dict):
        return data

    result = dict(data)
    for key in ("data", "subtypes"):
        if key in result and isinstance(result[key], list):
            result[key] = filter_array_records(
                result[key], keep_secact, keep_cancer_types=keep_cancer_types
            )
    if "proportions" in result and isinstance(result["proportions"], list):
        result["proportions"] = [
            r for r in result["proportions"]
            if not r.get("cancer_type") or r["cancer_type"] in keep_cancer_types
        ]
    if "secact_signatures" in result:
        result["secact_signatures"] = filter_signature_list(result["secact_signatures"], keep_secact)
    if "signatures" in result:
        result["signatures"] = [s for s in result["signatures"] if s in keep_secact or s not in (result.get("secact_signatures") or [])]
    return result


def process_exhaustion(data, keep_secact, keep_cancer_types):
    """Filter exhaustion.json."""
    if not isinstance(data, dict):
        return data

    result = dict(data)
    for key in ("data", "comparison", "by_subset"):
        if key in result and isinstance(result[key], list):
            result[key] = filter_array_records(
                result[key], keep_secact, keep_cancer_types=keep_cancer_types
            )
    if "secact_signatures" in result:
        result["secact_signatures"] = filter_signature_list(result["secact_signatures"], keep_secact)
    return result


def process_immune_infiltration(data, keep_secact, keep_cancer_types):
    """Filter immune_infiltration.json."""
    if not isinstance(data, dict):
        return data

    result = dict(data)
    for key in ("data",):
        if key in result and isinstance(result[key], list):
            result[key] = filter_array_records(
                result[key], keep_secact, keep_cancer_types=keep_cancer_types
            )
    # composition, tme_summary, by_study, by_donor: filter by cancer type
    for key in ("composition", "tme_summary", "by_study", "by_donor", "studies"):
        if key in result and isinstance(result[key], list):
            result[key] = [
                r for r in result[key]
                if not r.get("cancer_type") or r["cancer_type"] in keep_cancer_types
            ]
    if "secact_signatures" in result:
        result["secact_signatures"] = filter_signature_list(result["secact_signatures"], keep_secact)
    if "signatures" in result:
        # Keep CytoSig + filtered SecAct
        all_secact = set(data.get("secact_signatures", []))
        result["signatures"] = [s for s in result["signatures"] if s not in all_secact or s in keep_secact]
    if "cancer_types" in result:
        result["cancer_types"] = [ct for ct in result["cancer_types"] if ct in keep_cancer_types]
    return result


def process_inflammation_severity(data, keep_secact, keep_diseases):
    """Filter inflammation_severity.json or inflammation_severity_filtered.json."""
    if isinstance(data, list):
        return filter_array_records(data, keep_secact, keep_diseases=keep_diseases)
    return data


def process_inflammation_celltype_correlations(data, keep_secact, keep_celltypes):
    """Filter inflammation_celltype_correlations.json (dict with age/bmi arrays)."""
    if not isinstance(data, dict):
        return data

    result = {}
    ct_set = keep_celltypes.get("inflammation", set())
    for key in ("age", "bmi"):
        if key not in data:
            continue
        filtered = []
        for rec in data[key]:
            if not should_keep_signature(rec, keep_secact):
                continue
            ct = rec.get("cell_type", "")
            if ct and ct_set and ct not in ct_set:
                continue
            filtered.append(rec)
        result[key] = filtered
    return result


def process_gene_expression(data, keep_celltypes):
    """Filter gene_expression.json by cell type (14 MB array)."""
    if not isinstance(data, list):
        return data
    # Combine all kept cell types
    all_kept = set()
    for cts in keep_celltypes.values():
        all_kept.update(cts)
    return [r for r in data if not r.get("cell_type") or r["cell_type"] in all_kept]


def process_expression_boxplot(data, keep_celltypes):
    """Filter expression_boxplot.json by cell type (9 MB array)."""
    if not isinstance(data, list):
        return data
    all_kept = set()
    for cts in keep_celltypes.values():
        all_kept.update(cts)
    return [r for r in data if not r.get("cell_type") or r["cell_type"] in all_kept]


def process_cima_eqtl(data, keep_celltypes):
    """Filter cima_eqtl.json - keep structure, filter eqtls by cell type."""
    if not isinstance(data, dict):
        return data
    result = dict(data)
    ct_set = keep_celltypes.get("cima", set())
    if "eqtls" in result and isinstance(result["eqtls"], list) and ct_set:
        result["eqtls"] = [r for r in result["eqtls"] if not r.get("celltype") or r["celltype"] in ct_set]
    if "cell_types" in result and ct_set:
        result["cell_types"] = [ct for ct in result["cell_types"] if ct in ct_set]
    # Update summary
    if "summary" in result and isinstance(result["summary"], dict):
        summary = dict(result["summary"])
        summary["n_cell_types"] = len(result.get("cell_types", []))
        summary["total_eqtls"] = len(result.get("eqtls", []))
        result["summary"] = summary
    return result


def process_cross_atlas(data, keep_secact, keep_celltypes, keep_organs):
    """Filter cross_atlas.json - keep structure, filter large arrays."""
    if not isinstance(data, dict):
        return data

    result = dict(data)

    # conserved.signatures: filter by signature
    if "conserved" in result:
        conserved = dict(result["conserved"])
        if "signatures" in conserved:
            conserved["signatures"] = [
                s for s in conserved["signatures"]
                if should_keep_signature(s, keep_secact)
            ]
        result["conserved"] = conserved

    # meta_analysis: filter age/bmi/sex arrays
    if "meta_analysis" in result:
        ma = dict(result["meta_analysis"])
        for key in ("age", "bmi", "sex"):
            if key in ma and isinstance(ma[key], list):
                ma[key] = [r for r in ma[key] if should_keep_signature(r, keep_secact)]
        result["meta_analysis"] = ma

    # correlation: filter if present
    if "correlation" in result and isinstance(result["correlation"], dict):
        corr = dict(result["correlation"])
        if "secact" in corr and isinstance(corr["secact"], list):
            corr["secact"] = [
                r for r in corr["secact"]
                if r.get("signature", r.get("sig1", "")) in keep_secact
            ]
        result["correlation"] = corr

    # signature_reliability.secact: filter
    if "signature_reliability" in result and isinstance(result["signature_reliability"], dict):
        sr = dict(result["signature_reliability"])
        if "secact" in sr and isinstance(sr["secact"], dict):
            secact_rel = dict(sr["secact"])
            if "signatures" in secact_rel and isinstance(secact_rel["signatures"], list):
                secact_rel["signatures"] = [
                    s for s in secact_rel["signatures"]
                    if s.get("signature", "") in keep_secact
                ]
            sr["secact"] = secact_rel
        result["signature_reliability"] = sr

    # atlas_comparison.secact: filter
    if "atlas_comparison" in result and isinstance(result["atlas_comparison"], dict):
        ac = dict(result["atlas_comparison"])
        if "secact" in ac and isinstance(ac["secact"], dict):
            secact_ac = dict(ac["secact"])
            for pair_key, pair_data in secact_ac.items():
                if isinstance(pair_data, dict) and "data" in pair_data:
                    pd_copy = dict(pair_data)
                    pd_copy["data"] = [
                        r for r in pair_data["data"]
                        if r.get("signature", "") in keep_secact
                    ]
                    secact_ac[pair_key] = pd_copy
            ac["secact"] = secact_ac
        result["atlas_comparison"] = ac

    return result


def process_search_index(data, keep_secact):
    """Filter search_index.json - keep CytoSig + top SecAct entries."""
    if isinstance(data, list):
        return [r for r in data if should_keep_signature(r, keep_secact)]
    if isinstance(data, dict):
        result = dict(data)
        for key, val in result.items():
            if isinstance(val, list):
                result[key] = [
                    r for r in val
                    if not isinstance(r, dict) or should_keep_signature(r, keep_secact)
                ]
        return result
    return data


# ---------------------------------------------------------------------------
# Boxplot directory handling
# ---------------------------------------------------------------------------

def get_boxplot_files_to_keep(source_dir: Path, keep_secact: set) -> list:
    """Determine which boxplot files to keep."""
    boxplot_dir = source_dir / "boxplot"
    if not boxplot_dir.exists():
        return []

    keep = []
    for f in sorted(boxplot_dir.iterdir()):
        if not f.name.endswith("_activity.json"):
            continue
        sig_name = f.name.replace("_activity.json", "")
        # CytoSig signatures: always keep (they have specific known names)
        # We detect CytoSig by checking if there are ~43 short-named files
        # Simpler: keep if in keep_secact, or if the file is small enough
        # to be CytoSig (CytoSig files tend to be ~675KB vs SecAct ~340KB)
        # Actually simplest: keep all files whose signature is in either
        # the CytoSig set or keep_secact
        keep.append(f)  # We'll filter below

    return keep


def copy_boxplot_files(source_dir: Path, output_dir: Path, keep_secact: set, cytosig_sigs: set, dry_run: bool):
    """Copy matching boxplot files."""
    boxplot_src = source_dir / "boxplot"
    boxplot_dst = output_dir / "boxplot"

    if not boxplot_src.exists():
        print("  No boxplot/ directory found, skipping")
        return 0, 0

    if not dry_run:
        boxplot_dst.mkdir(parents=True, exist_ok=True)

    kept = 0
    total_size = 0
    for f in sorted(boxplot_src.iterdir()):
        if not f.name.endswith("_activity.json"):
            continue
        sig_name = f.name.replace("_activity.json", "")
        if sig_name in cytosig_sigs or sig_name in keep_secact:
            if not dry_run:
                shutil.copy2(f, boxplot_dst / f.name)
            kept += 1
            total_size += f.stat().st_size

    return kept, total_size


# ---------------------------------------------------------------------------
# Embedded data rebuild
# ---------------------------------------------------------------------------

def rebuild_embedded_data(lite_dir: Path):
    """Rebuild embedded_data.js from lite JSON files.

    Replicates the file-to-key mapping from 06_preprocess_viz_data.py.
    """
    print("\n  Rebuilding embedded_data.js from lite files...")

    json_files = [
        "cima_correlations.json",
        "cima_metabolites_top.json",
        "cima_differential.json",
        "cima_celltype.json",
        "cima_celltype_correlations.json",
        "cima_biochem_scatter.json",
        "cima_population_stratification.json",
        "cima_eqtl.json",
        "scatlas_organs.json",
        "scatlas_organs_top.json",
        "scatlas_celltypes.json",
        "cancer_comparison.json",
        "cancer_types.json",
        "immune_infiltration.json",
        "exhaustion.json",
        "caf_signatures.json",
        "organ_cancer_matrix.json",
        "adjacent_tissue.json",
        "inflammation_celltype.json",
        "inflammation_correlations.json",
        "inflammation_celltype_correlations.json",
        ("inflammation_disease_filtered.json", "inflammationdisease"),
        ("inflammation_severity_filtered.json", "inflammationseverity"),
        "inflammation_differential.json",
        "inflammation_longitudinal.json",
        "inflammation_cell_drivers.json",
        "inflammation_demographics.json",
        "disease_sankey.json",
        ("age_bmi_boxplots_filtered.json", "agebmiboxplots"),
        "treatment_response.json",
        "cohort_validation.json",
        "cross_atlas.json",
        "summary_stats.json",
    ]

    embedded = {}
    for item in json_files:
        if isinstance(item, tuple):
            json_file, key = item
        else:
            json_file = item
            key = json_file.replace(".json", "").replace("_", "")

        filepath = lite_dir / json_file
        if filepath.exists():
            with open(filepath) as f:
                embedded[key] = json.load(f)
            size_kb = filepath.stat().st_size / 1024
            print(f"    Embedded {json_file}: {size_kb:.1f} KB")
        else:
            print(f"    Skipping {json_file} (not found in lite dir)")

    js_content = f"const EMBEDDED_DATA = {json.dumps(embedded, separators=(',', ':'))};\n"

    js_path = lite_dir / "embedded_data.js"
    with open(js_path, "w") as f:
        f.write(js_content)

    total_size = js_path.stat().st_size / (1024 * 1024)
    print(f"    Total embedded_data.js: {total_size:.2f} MB")
    return embedded


# ---------------------------------------------------------------------------
# Get CytoSig signature names
# ---------------------------------------------------------------------------

def get_cytosig_signatures(source_dir: Path) -> set:
    """Extract CytoSig signature names from data files."""
    # Try cross_atlas.json first
    cross_atlas_path = source_dir / "cross_atlas.json"
    if cross_atlas_path.exists():
        with open(cross_atlas_path) as f:
            data = json.load(f)
        conserved = data.get("conserved", {}).get("signatures", [])
        cytosig = {s["signature"] for s in conserved if s.get("signature_type") == "CytoSig"}
        if cytosig:
            return cytosig

    # Fallback: try activity_boxplot or scatlas_celltypes
    for fname in ("scatlas_celltypes.json", "age_bmi_boxplots.json"):
        path = source_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "cytosig_signatures" in data:
            return set(data["cytosig_signatures"])
        # Nested check for age_bmi_boxplots
        for atlas_key in ("cima", "inflammation"):
            if atlas_key in data and "cytosig_signatures" in data[atlas_key]:
                return set(data[atlas_key]["cytosig_signatures"])

    return set()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        print(f"ERROR: Source directory does not exist: {source}")
        sys.exit(1)

    # Check available memory for the 5.5 GB file
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024**3)
        if avail_gb < 14:
            print(f"WARNING: Only {avail_gb:.1f} GB RAM available. "
                  f"bulk_donor_correlations.json needs ~14 GB. "
                  f"Consider running on HPC with more memory.")
    except ImportError:
        pass

    print("=" * 70)
    print("  CytoAtlas Data Lite Generator")
    print("=" * 70)
    print(f"  Source:  {source}")
    print(f"  Output:  {output}")
    print(f"  SecAct top-N: {args.secact_top_n}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Compute global filter sets
    # -----------------------------------------------------------------------
    print("\n[1/4] Computing global filter sets...")

    cytosig_sigs = get_cytosig_signatures(source)
    print(f"  CytoSig signatures: {len(cytosig_sigs)}")

    keep_secact = compute_secact_top_n(source, args.secact_top_n)

    # Include CytoSig names in SecAct/LinCytoSig filtering so the same
    # signature can be compared across signature types (cross-signature comparison)
    keep_secact |= cytosig_sigs

    keep_inflam_cts = compute_inflammation_celltypes(source, max_n=20)
    keep_scatlas_cts = compute_scatlas_celltypes(source, max_n=30)

    keep_cancer = compute_cancer_types_from_data(source, max_n=10)

    keep_celltypes = {
        "cima": KEEP_CIMA_CELLTYPES,
        "inflammation": keep_inflam_cts,
        "scatlas": keep_scatlas_cts,
    }

    print(f"\n  Filter summary:")
    print(f"    CytoSig signatures: {len(cytosig_sigs)} (keep all)")
    print(f"    SecAct signatures:  {len(keep_secact)} (of ~1,170, includes {len(cytosig_sigs)} CytoSig names for cross-comparison)")
    print(f"    CIMA cell types:    {len(KEEP_CIMA_CELLTYPES)} (of 27)")
    print(f"    Inflam cell types:  {len(keep_inflam_cts)} (of 66)")
    print(f"    scAtlas cell types: {len(keep_scatlas_cts)} (of 376)")
    print(f"    scAtlas organs:     {len(KEEP_ORGANS)} (of 35)")
    print(f"    Diseases:           {len(KEEP_DISEASES)} (of 20)")
    print(f"    Cancer types:       {len(keep_cancer)} (of 29)")

    if args.dry_run:
        print("\n  DRY RUN - no files will be written")
        return

    # -----------------------------------------------------------------------
    # Step 2: Create output directory
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Creating output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation").mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 3: Process each file
    # -----------------------------------------------------------------------
    print("\n[3/4] Processing files...")

    size_report = []  # (filename, orig_size, lite_size)
    total_orig = 0
    total_lite = 0

    # Build the processing plan
    files = sorted(f for f in source.iterdir() if f.is_file() and f.name != ".gitkeep")

    for filepath in files:
        fname = filepath.name
        orig_size = filepath.stat().st_size
        total_orig += orig_size

        # Skip embedded_data.js and embedded_data_lite.js - will rebuild
        if fname in ("embedded_data.js", "embedded_data_lite.js"):
            print(f"  SKIP {fname} (will rebuild)")
            continue

        # Small files: copy as-is
        if orig_size < SMALL_FILE_THRESHOLD:
            print(f"  COPY {fname} ({orig_size / 1024 / 1024:.2f} MB)")
            shutil.copy2(filepath, output / fname)
            lite_size = orig_size
            size_report.append((fname, orig_size, lite_size))
            total_lite += lite_size
            continue

        # Large files: load, filter, write
        print(f"  PROCESS {fname} ({orig_size / 1024 / 1024:.1f} MB)...", end="", flush=True)
        t0 = time.time()

        with open(filepath) as f:
            data = json.load(f)

        # Route to appropriate processor
        if fname == "activity_boxplot.json":
            result = process_activity_boxplot(data, keep_secact, keep_celltypes)
        elif fname in ("inflammation_disease.json", "inflammation_disease_filtered.json"):
            result = process_inflammation_disease(data, keep_secact, keep_celltypes, KEEP_DISEASES)
        elif fname == "singlecell_activity.json":
            result = process_singlecell_activity(data, keep_secact, keep_celltypes)
        elif fname == "scatlas_celltypes.json":
            result = process_scatlas_celltypes(data, keep_secact, KEEP_ORGANS, keep_scatlas_cts)
        elif fname in ("age_bmi_boxplots.json", "age_bmi_boxplots_filtered.json"):
            result = process_age_bmi_boxplots(data, keep_secact, keep_celltypes)
        elif fname in ("scatlas_normal_singlecell_secact.json", "scatlas_cancer_singlecell_secact.json"):
            result = process_scatlas_singlecell(data, keep_secact, KEEP_ORGANS, keep_scatlas_cts)
        elif fname == "bulk_donor_correlations.json":
            result = process_bulk_donor_correlations(data, keep_secact)
        elif fname == "bulk_rnaseq_validation.json":
            result = process_bulk_rnaseq_validation(data, keep_secact)
        elif fname == "caf_signatures.json":
            result = process_caf_signatures(data, keep_secact, keep_cancer)
        elif fname == "exhaustion.json":
            result = process_exhaustion(data, keep_secact, keep_cancer)
        elif fname == "immune_infiltration.json":
            result = process_immune_infiltration(data, keep_secact, keep_cancer)
        elif fname in ("inflammation_severity.json", "inflammation_severity_filtered.json"):
            result = process_inflammation_severity(data, keep_secact, KEEP_DISEASES)
        elif fname == "inflammation_celltype_correlations.json":
            result = process_inflammation_celltype_correlations(data, keep_secact, keep_celltypes)
        elif fname == "gene_expression.json":
            result = process_gene_expression(data, keep_celltypes)
        elif fname == "expression_boxplot.json":
            result = process_expression_boxplot(data, keep_celltypes)
        elif fname == "cima_eqtl.json":
            result = process_cima_eqtl(data, keep_celltypes)
        elif fname == "cross_atlas.json":
            result = process_cross_atlas(data, keep_secact, keep_celltypes, KEEP_ORGANS)
        elif fname == "search_index.json":
            result = process_search_index(data, keep_secact)
        elif fname == "cima_celltype_correlations.json":
            # Filter by signature and CIMA cell types
            if isinstance(data, dict):
                result = {}
                for key, records in data.items():
                    if isinstance(records, list):
                        result[key] = [r for r in records if should_keep_signature(r, keep_secact)]
                    else:
                        result[key] = records
            else:
                result = filter_array_records(data, keep_secact)
        else:
            # Generic: try to filter if it's an array or dict with arrays
            if isinstance(data, list):
                result = filter_array_records(data, keep_secact)
            elif isinstance(data, dict):
                # Try to filter any list values
                result = {}
                for key, val in data.items():
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        result[key] = filter_array_records(val, keep_secact)
                    else:
                        result[key] = val
            else:
                result = data

        # Free the original data
        del data

        # Write with compact JSON
        out_path = output / fname
        with open(out_path, "w") as f:
            json.dump(result, f, separators=(",", ":"))

        del result

        lite_size = out_path.stat().st_size
        elapsed = time.time() - t0
        reduction = (1 - lite_size / orig_size) * 100 if orig_size > 0 else 0
        print(f" {lite_size / 1024 / 1024:.1f} MB ({reduction:.0f}% reduction, {elapsed:.1f}s)")

        size_report.append((fname, orig_size, lite_size))
        total_lite += lite_size

    # Process validation/ subdirectory
    val_dir = source / "validation"
    if val_dir.exists():
        print("\n  Processing validation/ files...")
        for vf in sorted(val_dir.iterdir()):
            if not vf.name.endswith(".json"):
                continue
            orig_size = vf.stat().st_size
            total_orig += orig_size
            print(f"  PROCESS validation/{vf.name} ({orig_size / 1024 / 1024:.1f} MB)...", end="", flush=True)
            t0 = time.time()

            with open(vf) as f:
                data = json.load(f)

            result = process_validation_file(data, keep_secact)
            del data

            out_path = output / "validation" / vf.name
            with open(out_path, "w") as f:
                json.dump(result, f, separators=(",", ":"))
            del result

            lite_size = out_path.stat().st_size
            elapsed = time.time() - t0
            reduction = (1 - lite_size / orig_size) * 100 if orig_size > 0 else 0
            print(f" {lite_size / 1024 / 1024:.1f} MB ({reduction:.0f}% reduction, {elapsed:.1f}s)")
            size_report.append((f"validation/{vf.name}", orig_size, lite_size))
            total_lite += lite_size

    # Process boxplot/ subdirectory
    print("\n  Processing boxplot/ files...")
    kept_boxplots, boxplot_size = copy_boxplot_files(
        source, output, keep_secact, cytosig_sigs, args.dry_run
    )
    # Get original boxplot size
    boxplot_src = source / "boxplot"
    orig_boxplot_size = sum(f.stat().st_size for f in boxplot_src.iterdir() if f.is_file()) if boxplot_src.exists() else 0
    total_orig += orig_boxplot_size
    total_lite += boxplot_size
    print(f"  Kept {kept_boxplots} of 1191 boxplot files ({boxplot_size / 1024 / 1024:.1f} MB)")
    size_report.append(("boxplot/ (directory)", orig_boxplot_size, boxplot_size))

    # -----------------------------------------------------------------------
    # Step 3b: Rebuild embedded_data.js
    # -----------------------------------------------------------------------
    rebuild_embedded_data(output)
    embedded_size = (output / "embedded_data.js").stat().st_size if (output / "embedded_data.js").exists() else 0
    total_lite += embedded_size
    size_report.append(("embedded_data.js (rebuilt)", 0, embedded_size))

    # -----------------------------------------------------------------------
    # Step 4: Print report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Size Comparison Report")
    print("=" * 70)
    print(f"  {'File':<50} {'Original':>10} {'Lite':>10} {'Reduction':>10}")
    print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*10}")

    for fname, orig, lite in sorted(size_report, key=lambda x: -x[1]):
        orig_mb = orig / (1024 * 1024)
        lite_mb = lite / (1024 * 1024)
        reduction = (1 - lite / orig) * 100 if orig > 0 else 0
        if orig_mb >= 1:
            print(f"  {fname:<50} {orig_mb:>9.1f}M {lite_mb:>9.1f}M {reduction:>9.0f}%")

    print(f"\n  {'TOTAL':<50} {total_orig / (1024**3):>9.2f}G {total_lite / (1024**2):>9.1f}M")
    overall_reduction = (1 - total_lite / total_orig) * 100 if total_orig > 0 else 0
    print(f"  Overall reduction: {overall_reduction:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
