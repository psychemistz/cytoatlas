#!/usr/bin/env python3
"""Generate celltype-only scatter data for the Cell Type Level validation tab.

Reads celltype-only pseudobulk expression and activity H5AD files
(cells pooled by celltype, ignoring donor) and produces scatter JSON files
in the same format as existing celltype_scatter files.

Each point in the scatter = one cell type (all donors pooled).

Usage:
    python scripts/17_generate_celltype_only_scatter.py
    python scripts/17_generate_celltype_only_scatter.py --duckdb  # also write to DuckDB
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATLAS_VALIDATION_DIR = PROJECT_ROOT / "results" / "atlas_validation"
OUTPUT_DIR = PROJECT_ROOT / "visualization" / "data" / "validation" / "celltype_scatter"
MAPPING_PATH = PROJECT_ROOT / "cytoatlas-api" / "static" / "data" / "signature_gene_mapping.json"

SIG_TYPES = ["cytosig", "lincytosig", "secact"]

# Celltype-only H5AD patterns: atlas -> [(level_name, pseudobulk_file, activity_prefix)]
# pseudobulk: {atlas_validation_dir}/{atlas}/pseudobulk/{pseudobulk_file}
# activity:   {atlas_validation_dir}/{atlas}/activity/{activity_prefix}_{sig}.h5ad
CELLTYPE_ONLY_PATTERNS = {
    "cima": [
        ("l1", "cima_pseudobulk_l1.h5ad", "cima_l1"),
        ("l2", "cima_pseudobulk_l2.h5ad", "cima_l2"),
        ("l3", "cima_pseudobulk_l3.h5ad", "cima_l3"),
        ("l4", "cima_pseudobulk_l4.h5ad", "cima_l4"),
    ],
    "inflammation_main": [
        ("l1", "inflammation_main_pseudobulk_l1.h5ad", "inflammation_main_l1"),
        ("l2", "inflammation_main_pseudobulk_l2.h5ad", "inflammation_main_l2"),
    ],
    "inflammation_val": [
        ("l1", "inflammation_val_pseudobulk_l1.h5ad", "inflammation_val_l1"),
        ("l2", "inflammation_val_pseudobulk_l2.h5ad", "inflammation_val_l2"),
    ],
    "inflammation_ext": [
        ("l1", "inflammation_ext_pseudobulk_l1.h5ad", "inflammation_ext_l1"),
        ("l2", "inflammation_ext_pseudobulk_l2.h5ad", "inflammation_ext_l2"),
    ],
    "scatlas_normal": [
        ("celltype", "scatlas_normal_pseudobulk_celltype.h5ad", "scatlas_normal_celltype"),
        ("organ_celltype", "scatlas_normal_pseudobulk_organ_celltype.h5ad", "scatlas_normal_organ_celltype"),
    ],
    "scatlas_cancer": [
        ("celltype", "scatlas_cancer_pseudobulk_celltype.h5ad", "scatlas_cancer_celltype"),
        ("organ_celltype", "scatlas_cancer_pseudobulk_organ_celltype.h5ad", "scatlas_cancer_organ_celltype"),
    ],
}


def round_val(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), decimals)


def fisher_z_ci(rho, n, z_crit=1.96):
    if n < 6 or rho is None or not np.isfinite(rho) or abs(rho) >= 1.0:
        return None
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    return [round_val(np.tanh(z - z_crit * se)), round_val(np.tanh(z + z_crit * se))]


def load_gene_mapping():
    if not MAPPING_PATH.exists():
        return {}
    with open(MAPPING_PATH) as f:
        data = json.load(f)
    mapping = {}
    for target_name, info in data.get("cytosig_mapping", {}).items():
        mapping[target_name] = info["hgnc_symbol"]
    return mapping


def resolve_gene(target, gene_set, cytosig_map, symbol_to_varname):
    """Resolve target name to gene symbol and var_name in expression matrix."""
    # Direct match
    if target in gene_set:
        return target, symbol_to_varname.get(target, target)
    # CytoSig mapping
    if target in cytosig_map:
        mapped = cytosig_map[target]
        if mapped in gene_set:
            return mapped, symbol_to_varname.get(mapped, mapped)
    # LinCytoSig: celltype__TARGET format
    if "__" in target:
        cytokine = target.split("__")[-1]
        if cytokine in gene_set:
            return cytokine, symbol_to_varname.get(cytokine, cytokine)
        if cytokine in cytosig_map:
            mapped = cytosig_map[cytokine]
            if mapped in gene_set:
                return mapped, symbol_to_varname.get(mapped, mapped)
    return None, None


def generate_scatter_for_level(atlas, level_name, expr_path, act_path, cytosig_map):
    """Generate scatter data for one atlas/level/sigtype combination.

    Returns dict: {target: {gene, rho, pval, n, rho_ci, celltypes, groups, points}}
    """
    expr_adata = ad.read_h5ad(expr_path)
    act_adata = ad.read_h5ad(act_path)

    # Build gene lookup
    gene_set = set(expr_adata.var_names)
    symbol_to_varname = {}
    if "symbol" in expr_adata.var.columns:
        for idx, sym in zip(expr_adata.var_names, expr_adata.var["symbol"]):
            if pd.notna(sym):
                symbol_to_varname[str(sym)] = idx
        gene_set = set(symbol_to_varname.keys()) | gene_set

    # Find celltype column
    ct_col = None
    for col in ["cell_type", "celltype", "cell_type_l1", "cell_type_l2",
                 "cell_type_l3", "cell_type_l4"]:
        if col in expr_adata.obs.columns:
            ct_col = col
            break

    common_obs = expr_adata.obs_names.intersection(act_adata.obs_names)
    if len(common_obs) < 3:
        print(f"    WARNING: Only {len(common_obs)} common obs, skipping")
        return {}

    # Get celltype labels (each obs IS a celltype in celltype-only data)
    if ct_col:
        celltypes_available = sorted(
            expr_adata.obs.loc[common_obs, ct_col].dropna().unique().tolist()
        )
        ct_labels = expr_adata.obs.loc[common_obs, ct_col].values
    else:
        # Use obs index as celltype name
        celltypes_available = sorted(common_obs.tolist())
        ct_labels = np.array(common_obs.tolist())

    ct_to_idx = {ct: i for i, ct in enumerate(celltypes_available)}

    targets_data = {}

    for target in act_adata.var_names:
        gene, expr_var = resolve_gene(target, gene_set, cytosig_map, symbol_to_varname)
        if gene is None or expr_var not in set(expr_adata.var_names):
            continue

        expr_vals = expr_adata[common_obs, expr_var].X.flatten()
        act_vals = act_adata[common_obs, target].X.flatten()
        if hasattr(expr_vals, "toarray"):
            expr_vals = expr_vals.toarray().flatten()
        if hasattr(act_vals, "toarray"):
            act_vals = act_vals.toarray().flatten()

        ct_arr = ct_labels if ct_col else np.array(common_obs.tolist())

        # Remove NaN pairs
        mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
        expr_vals = expr_vals[mask].astype(np.float64)
        act_vals = act_vals[mask].astype(np.float64)
        ct_arr = ct_arr[mask]

        n_total = len(expr_vals)
        if n_total < 3:
            continue

        rho, pval = stats.spearmanr(expr_vals, act_vals)

        # Z-score normalize expression
        expr_std = np.std(expr_vals)
        if expr_std > 1e-10:
            expr_z = (expr_vals - np.mean(expr_vals)) / expr_std
        else:
            expr_z = expr_vals

        rho_ci = fisher_z_ci(rho, n_total)

        points = []
        for i in range(n_total):
            pt = [round_val(float(expr_z[i]), 2), round_val(float(act_vals[i]), 2)]
            pt.append(ct_to_idx.get(str(ct_arr[i]), -1))
            points.append(pt)

        targets_data[target] = {
            "gene": gene,
            "rho": round_val(rho),
            "pval": float(f"{pval:.2e}") if pval is not None and np.isfinite(pval) else None,
            "n": n_total,
            "rho_ci": rho_ci,
            "celltypes": celltypes_available,
            "groups": celltypes_available,
            "points": points,
        }

    del expr_adata, act_adata
    gc.collect()

    return targets_data


def write_duckdb(all_data, db_path):
    """Write scatter data to DuckDB table."""
    try:
        import duckdb
    except ImportError:
        print("WARNING: duckdb not installed, skipping DuckDB output")
        return

    rows = []
    for atlas, levels in all_data.items():
        for level, sigs in levels.items():
            for sig_type, targets in sigs.items():
                for target, info in targets.items():
                    for pt in info["points"]:
                        group_name = info["groups"][pt[2]] if pt[2] >= 0 and pt[2] < len(info["groups"]) else ""
                        rows.append({
                            "atlas": atlas,
                            "level": level,
                            "sig_type": sig_type,
                            "target": target,
                            "gene": info["gene"],
                            "celltype": group_name,
                            "expression_z": pt[0],
                            "activity": pt[1],
                            "rho": info["rho"],
                            "pval": info["pval"],
                            "n": info["n"],
                        })

    if not rows:
        print("No data to write to DuckDB")
        return

    df = pd.DataFrame(rows)
    db = duckdb.connect(str(db_path))
    db.execute("DROP TABLE IF EXISTS celltype_only_scatter")
    db.execute("CREATE TABLE celltype_only_scatter AS SELECT * FROM df")
    db.execute("CREATE INDEX idx_ctos_atlas ON celltype_only_scatter(atlas, level, sig_type)")
    db.execute("CREATE INDEX idx_ctos_target ON celltype_only_scatter(atlas, level, sig_type, target)")
    n = db.execute("SELECT count(*) FROM celltype_only_scatter").fetchone()[0]
    print(f"\nDuckDB: wrote {n:,} rows to celltype_only_scatter in {db_path}")
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Generate celltype-only scatter data")
    parser.add_argument("--duckdb", action="store_true", help="Also write to DuckDB")
    parser.add_argument("--duckdb-path", type=str,
                        default=str(PROJECT_ROOT / "atlas_data.duckdb"),
                        help="DuckDB file path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cytosig_map = load_gene_mapping()

    all_data = {}  # {atlas: {level: {sig: {target: {...}}}}}
    total_files = 0

    for atlas, levels in CELLTYPE_ONLY_PATTERNS.items():
        atlas_dir = ATLAS_VALIDATION_DIR / atlas
        if not atlas_dir.exists():
            print(f"WARNING: {atlas_dir} not found, skipping {atlas}")
            continue

        atlas_data = {}

        for level_name, pb_file, act_prefix in levels:
            level_data = {}
            expr_path = atlas_dir / "pseudobulk" / pb_file

            if not expr_path.exists():
                print(f"  WARNING: {expr_path} not found, skipping {atlas}/{level_name}")
                continue

            for sig_type in SIG_TYPES:
                act_path = atlas_dir / "activity" / f"{act_prefix}_{sig_type}.h5ad"
                if not act_path.exists():
                    print(f"  WARNING: {act_path.name} not found, skipping")
                    continue

                print(f"  Processing {atlas}/{level_name}/{sig_type}...")
                targets = generate_scatter_for_level(
                    atlas, level_name, expr_path, act_path, cytosig_map
                )

                if targets:
                    # Write JSON
                    out_file = OUTPUT_DIR / f"{atlas}_{level_name}_{sig_type}.json"
                    with open(out_file, "w") as f:
                        json.dump(targets, f, separators=(",", ":"))
                    size_mb = out_file.stat().st_size / (1024 * 1024)
                    print(f"    -> {out_file.name}: {len(targets)} targets ({size_mb:.1f} MB)")
                    total_files += 1
                    level_data[sig_type] = targets

            atlas_data[level_name] = level_data
        all_data[atlas] = atlas_data

    print(f"\nGenerated {total_files} scatter JSON files in {OUTPUT_DIR}")

    if args.duckdb:
        write_duckdb(all_data, args.duckdb_path)

    # Print summary
    print("\n=== Summary ===")
    for atlas, levels in all_data.items():
        for level, sigs in levels.items():
            for sig, targets in sigs.items():
                if targets:
                    first_t = next(iter(targets.values()))
                    print(f"  {atlas}/{level}/{sig}: {len(targets)} targets, {first_t['n']} celltypes/points")


if __name__ == "__main__":
    main()
