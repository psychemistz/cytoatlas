#!/usr/bin/env python3
"""Generate cell-type specific SecAct boxplot data for Inflammation atlas."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/visualization/data")
INFLAM_DIR = Path("/vf/users/parks34/projects/2secactpy/results/inflammation")
META_PATH = Path("/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv")


def generate_celltype_boxplots():
    """Generate cell-type specific boxplots for all SecAct signatures."""
    print("Loading sample metadata...")
    meta = pd.read_csv(META_PATH)
    meta = meta.rename(columns={"sampleID": "sample"})

    # Map binned_age to standard bins
    age_mapping = {
        '<18': '<30', '18-30': '<30',
        '31-40': '30-39', '41-50': '40-49',
        '51-60': '50-59', '61-70': '60-69',
        '71-80': '70+', '>80': '70+'
    }
    meta["age_bin"] = meta["binned_age"].map(age_mapping)

    # Create BMI bins
    meta["bmi_bin"] = pd.cut(
        meta["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    print(f"Samples: {len(meta)}")
    print(f"Age bins: {meta['age_bin'].value_counts().to_dict()}")
    print(f"BMI bins (non-null): {meta['bmi_bin'].value_counts().to_dict()}")

    # Load SecAct pseudobulk data
    print("\nLoading SecAct pseudobulk data...")
    h5ad_path = INFLAM_DIR / "main_SecAct_pseudobulk.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    print(f"Shape: {adata.shape} (proteins × sample_celltype)")

    # Create activity DataFrame (proteins as rows, sample_celltype as columns)
    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    # Get sample info from h5ad
    sample_info = adata.var[["sample", "cell_type", "n_cells"]].copy()
    print(f"Unique samples: {sample_info['sample'].nunique()}")
    print(f"Unique cell types: {sample_info['cell_type'].nunique()}")
    cell_types = sorted(sample_info["cell_type"].unique())
    print(f"Cell types: {cell_types[:10]}... ({len(cell_types)} total)")

    # Merge sample metadata with h5ad sample info
    sample_info = sample_info.reset_index().rename(columns={"index": "column"})
    sample_info = sample_info.merge(meta[["sample", "age_bin", "bmi_bin"]], on="sample", how="left")
    sample_info = sample_info.set_index("column")

    sig_cols = list(activity_df.index)  # Protein/signature names
    print(f"Signatures: {len(sig_cols)}")

    age_bins = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    bmi_bins = ["Underweight", "Normal", "Overweight", "Obese"]

    # Generate cell-type specific age boxplots
    print("\nGenerating cell-type specific age boxplots...")
    age_data = []
    total_combos = len(sig_cols) * len(cell_types)
    processed = 0

    for sig in sig_cols:
        sig_values = activity_df.loc[sig]  # All values for this signature

        for ct in cell_types:
            processed += 1
            if processed % 10000 == 0:
                print(f"  Processed {processed}/{total_combos} combinations...")

            # Get columns for this cell type
            ct_mask = sample_info["cell_type"] == ct
            ct_cols = sample_info[ct_mask].index

            for bin_val in age_bins:
                # Get columns for this cell type AND age bin
                bin_mask = (sample_info["cell_type"] == ct) & (sample_info["age_bin"] == bin_val)
                bin_cols = sample_info[bin_mask].index

                if len(bin_cols) < 3:
                    continue

                bin_data = sig_values[bin_cols].dropna()
                if len(bin_data) < 3:
                    continue

                q1 = bin_data.quantile(0.25)
                q3 = bin_data.quantile(0.75)
                iqr = q3 - q1
                whisker_low = max(bin_data.min(), q1 - 1.5 * iqr)
                whisker_high = min(bin_data.max(), q3 + 1.5 * iqr)

                age_data.append({
                    "signature": sig,
                    "sig_type": "SecAct",
                    "bin": bin_val,
                    "cell_type": ct,
                    "min": round(float(whisker_low), 4),
                    "q1": round(float(q1), 4),
                    "median": round(float(bin_data.median()), 4),
                    "q3": round(float(q3), 4),
                    "max": round(float(whisker_high), 4),
                    "mean": round(float(bin_data.mean()), 4),
                    "n": len(bin_data),
                })

    print(f"Generated {len(age_data)} cell-type specific age boxplots")

    # Generate cell-type specific BMI boxplots
    print("\nGenerating cell-type specific BMI boxplots...")
    bmi_data = []
    processed = 0

    for sig in sig_cols:
        sig_values = activity_df.loc[sig]

        for ct in cell_types:
            processed += 1
            if processed % 10000 == 0:
                print(f"  Processed {processed}/{total_combos} combinations...")

            for bin_val in bmi_bins:
                bin_mask = (sample_info["cell_type"] == ct) & (sample_info["bmi_bin"] == bin_val)
                bin_cols = sample_info[bin_mask].index

                if len(bin_cols) < 3:
                    continue

                bin_data = sig_values[bin_cols].dropna()
                if len(bin_data) < 3:
                    continue

                q1 = bin_data.quantile(0.25)
                q3 = bin_data.quantile(0.75)
                iqr = q3 - q1
                whisker_low = max(bin_data.min(), q1 - 1.5 * iqr)
                whisker_high = min(bin_data.max(), q3 + 1.5 * iqr)

                bmi_data.append({
                    "signature": sig,
                    "sig_type": "SecAct",
                    "bin": bin_val,
                    "cell_type": ct,
                    "min": round(float(whisker_low), 4),
                    "q1": round(float(q1), 4),
                    "median": round(float(bin_data.median()), 4),
                    "q3": round(float(q3), 4),
                    "max": round(float(whisker_high), 4),
                    "mean": round(float(bin_data.mean()), 4),
                    "n": len(bin_data),
                })

    print(f"Generated {len(bmi_data)} cell-type specific BMI boxplots")

    return age_data, bmi_data


def main():
    # Load existing boxplot data
    print("Loading existing boxplot data...")
    boxplot_path = OUTPUT_DIR / "age_bmi_boxplots.json"
    with open(boxplot_path) as f:
        boxplot_data = json.load(f)

    # Generate new cell-type specific SecAct data
    new_age_data, new_bmi_data = generate_celltype_boxplots()

    # Remove old cell-type specific SecAct data
    print("\nReplacing old cell-type specific SecAct data...")
    old_age = boxplot_data["inflammation"]["age"]
    old_bmi = boxplot_data["inflammation"]["bmi"]

    # Keep sample-level SecAct and all CytoSig data
    filtered_age = [
        r for r in old_age
        if not (r.get("sig_type") == "SecAct" and r.get("cell_type") not in (None, "All"))
    ]
    filtered_bmi = [
        r for r in old_bmi
        if not (r.get("sig_type") == "SecAct" and r.get("cell_type") not in (None, "All"))
    ]

    print(f"Old age data: {len(old_age)}, filtered: {len(filtered_age)}")
    print(f"Old BMI data: {len(old_bmi)}, filtered: {len(filtered_bmi)}")

    # Add new cell-type specific SecAct data
    boxplot_data["inflammation"]["age"] = filtered_age + new_age_data
    boxplot_data["inflammation"]["bmi"] = filtered_bmi + new_bmi_data

    print(f"\nFinal age data: {len(boxplot_data['inflammation']['age'])}")
    print(f"Final BMI data: {len(boxplot_data['inflammation']['bmi'])}")

    # Save
    print(f"\nSaving to {boxplot_path}...")
    with open(boxplot_path, "w") as f:
        json.dump(boxplot_data, f)

    print("Done!")


if __name__ == "__main__":
    main()
