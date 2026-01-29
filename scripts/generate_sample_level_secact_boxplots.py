#!/usr/bin/env python3
"""Generate sample-level SecAct boxplot data for all 1,170 signatures."""

import json
from pathlib import Path

import anndata as ad
import pandas as pd

OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/visualization/data")
CIMA_DIR = Path("/vf/users/parks34/projects/2secactpy/results/cima")
CIMA_META_PATH = Path("/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv")


def generate_sample_level_boxplots():
    """Generate sample-level boxplots for all SecAct signatures."""
    print("Loading sample metadata...")
    cima_meta = pd.read_csv(CIMA_META_PATH)
    cima_meta = cima_meta.rename(columns={"Sample_name": "sample", "Age": "age"})

    # Create age bins
    cima_meta["age_bin"] = pd.cut(
        cima_meta["age"],
        bins=[0, 30, 40, 50, 60, 70, 100],
        labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"],
    )
    # Create BMI bins
    cima_meta["bmi_bin"] = pd.cut(
        cima_meta["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    print(f"Samples: {len(cima_meta)}")
    print(f"Age range: {cima_meta['age'].min()} - {cima_meta['age'].max()}")
    print(f"BMI range: {cima_meta['BMI'].min():.1f} - {cima_meta['BMI'].max():.1f}")

    # Load SecAct pseudobulk data
    print("\nLoading SecAct pseudobulk data...")
    h5ad_path = CIMA_DIR / "CIMA_SecAct_pseudobulk.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    print(f"Shape: {adata.shape} (proteins × sample_celltype)")

    # Create activity DataFrame (proteins as rows, sample_celltype as columns)
    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    # Get sample info
    sample_info = adata.var[["sample", "cell_type", "n_cells"]].copy()
    print(f"Unique samples: {sample_info['sample'].nunique()}")

    # Aggregate to sample level (weighted by n_cells)
    print("\nAggregating to sample level...")
    sample_activity = {}
    for sample in sample_info["sample"].unique():
        sample_cols = sample_info[sample_info["sample"] == sample].index
        weights = sample_info.loc[sample_cols, "n_cells"].values
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
            sample_activity[sample] = weighted_mean

    # Transpose: rows=samples, columns=proteins
    sample_activity_df = pd.DataFrame(sample_activity).T
    sample_activity_df = sample_activity_df.reset_index().rename(columns={"index": "sample"})

    # Merge with metadata
    sample_activity_df = sample_activity_df.merge(
        cima_meta[["sample", "age_bin", "bmi_bin"]], on="sample", how="left"
    )

    sig_cols = [c for c in sample_activity_df.columns if c not in ["sample", "age_bin", "bmi_bin"]]
    print(f"Signatures: {len(sig_cols)}")

    # Generate age boxplots
    print("\nGenerating age boxplots...")
    age_data = []
    age_bins = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    for i, sig in enumerate(sig_cols):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(sig_cols)} signatures...")
        for bin_val in age_bins:
            bin_data = sample_activity_df[sample_activity_df["age_bin"] == bin_val][sig].dropna()
            if len(bin_data) >= 3:
                q1 = bin_data.quantile(0.25)
                q3 = bin_data.quantile(0.75)
                iqr = q3 - q1
                whisker_low = max(bin_data.min(), q1 - 1.5 * iqr)
                whisker_high = min(bin_data.max(), q3 + 1.5 * iqr)

                age_data.append({
                    "signature": sig,
                    "sig_type": "SecAct",
                    "bin": bin_val,
                    "cell_type": None,  # Sample-level
                    "min": round(float(whisker_low), 4),
                    "q1": round(float(q1), 4),
                    "median": round(float(bin_data.median()), 4),
                    "q3": round(float(q3), 4),
                    "max": round(float(whisker_high), 4),
                    "mean": round(float(bin_data.mean()), 4),
                    "n": len(bin_data),
                })

    print(f"Generated {len(age_data)} age boxplots")

    # Generate BMI boxplots
    print("\nGenerating BMI boxplots...")
    bmi_data = []
    bmi_bins = ["Underweight", "Normal", "Overweight", "Obese"]
    for i, sig in enumerate(sig_cols):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(sig_cols)} signatures...")
        for bin_val in bmi_bins:
            bin_data = sample_activity_df[sample_activity_df["bmi_bin"] == bin_val][sig].dropna()
            if len(bin_data) >= 3:
                q1 = bin_data.quantile(0.25)
                q3 = bin_data.quantile(0.75)
                iqr = q3 - q1
                whisker_low = max(bin_data.min(), q1 - 1.5 * iqr)
                whisker_high = min(bin_data.max(), q3 + 1.5 * iqr)

                bmi_data.append({
                    "signature": sig,
                    "sig_type": "SecAct",
                    "bin": bin_val,
                    "cell_type": None,  # Sample-level
                    "min": round(float(whisker_low), 4),
                    "q1": round(float(q1), 4),
                    "median": round(float(bin_data.median()), 4),
                    "q3": round(float(q3), 4),
                    "max": round(float(whisker_high), 4),
                    "mean": round(float(bin_data.mean()), 4),
                    "n": len(bin_data),
                })

    print(f"Generated {len(bmi_data)} BMI boxplots")

    return age_data, bmi_data, sorted(sig_cols)


def main():
    # Load existing boxplot data
    print("Loading existing boxplot data...")
    boxplot_path = OUTPUT_DIR / "age_bmi_boxplots.json"
    with open(boxplot_path) as f:
        boxplot_data = json.load(f)

    # Generate new sample-level SecAct data
    new_age_data, new_bmi_data, secact_signatures = generate_sample_level_boxplots()

    # Remove old sample-level SecAct data (cell_type is None or 'All')
    print("\nReplacing old sample-level SecAct data...")
    old_age = boxplot_data["cima"]["age"]
    old_bmi = boxplot_data["cima"]["bmi"]

    # Keep non-SecAct sample-level and all cell-type specific data
    filtered_age = [
        r for r in old_age
        if not (r.get("sig_type") == "SecAct" and r.get("cell_type") in (None, "All"))
    ]
    filtered_bmi = [
        r for r in old_bmi
        if not (r.get("sig_type") == "SecAct" and r.get("cell_type") in (None, "All"))
    ]

    print(f"Old age data: {len(old_age)}, filtered: {len(filtered_age)}")
    print(f"Old BMI data: {len(old_bmi)}, filtered: {len(filtered_bmi)}")

    # Add new sample-level SecAct data
    boxplot_data["cima"]["age"] = filtered_age + new_age_data
    boxplot_data["cima"]["bmi"] = filtered_bmi + new_bmi_data
    boxplot_data["cima"]["secact_signatures"] = secact_signatures

    print(f"\nFinal age data: {len(boxplot_data['cima']['age'])}")
    print(f"Final BMI data: {len(boxplot_data['cima']['bmi'])}")
    print(f"SecAct signatures: {len(secact_signatures)}")

    # Save
    print(f"\nSaving to {boxplot_path}...")
    with open(boxplot_path, "w") as f:
        json.dump(boxplot_data, f)

    print("Done!")


if __name__ == "__main__":
    main()
