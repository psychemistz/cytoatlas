"""
LinCytoSig-specific validation.

Validates cell-type-specific cytokine signatures by checking that
activity correlates with target gene expression within the matching cell type.

For example, "Macrophage__IFNG" activity should correlate with IFNG expression
specifically in Macrophage cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LinCytoSigResult:
    """Result of LinCytoSig validation for a single signature."""

    signature: str
    """Full signature name (e.g., 'Macrophage__IFNG')."""

    cell_type: str
    """Cell type component."""

    cytokine: str
    """Cytokine component."""

    correlation: float
    """Pearson correlation between expression and activity."""

    spearman: float
    """Spearman correlation."""

    pvalue: float
    """P-value for Pearson correlation."""

    r2: float
    """R-squared."""

    n_samples: int
    """Number of samples/cells used."""

    mean_activity: float
    """Mean activity in this cell type."""

    mean_expression: float
    """Mean target gene expression in this cell type."""

    activity_rank: Optional[int] = None
    """Rank of this cell type's activity among all cell types (1=highest)."""

    expression_rank: Optional[int] = None
    """Rank of this cell type's expression among all cell types."""


@dataclass
class LinCytoSigValidationSummary:
    """Summary of LinCytoSig validation across all signatures."""

    results: list[LinCytoSigResult]
    """Individual signature results."""

    n_signatures: int = 0
    """Total signatures validated."""

    n_significant: int = 0
    """Signatures with p < 0.05."""

    n_positive: int = 0
    """Signatures with positive correlation."""

    mean_correlation: float = 0.0
    """Mean correlation across signatures."""

    median_correlation: float = 0.0
    """Median correlation."""

    mean_r2: float = 0.0
    """Mean R-squared."""

    def __post_init__(self):
        if self.results:
            self.n_signatures = len(self.results)
            self.n_significant = sum(1 for r in self.results if r.pvalue < 0.05)
            self.n_positive = sum(1 for r in self.results if r.correlation > 0)
            correlations = [r.correlation for r in self.results]
            self.mean_correlation = np.mean(correlations)
            self.median_correlation = np.median(correlations)
            self.mean_r2 = np.mean([r.r2 for r in self.results])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        records = []
        for r in self.results:
            records.append({
                "signature": r.signature,
                "cell_type": r.cell_type,
                "cytokine": r.cytokine,
                "correlation": r.correlation,
                "spearman": r.spearman,
                "pvalue": r.pvalue,
                "r2": r.r2,
                "n_samples": r.n_samples,
                "mean_activity": r.mean_activity,
                "mean_expression": r.mean_expression,
                "activity_rank": r.activity_rank,
                "expression_rank": r.expression_rank,
            })
        return pd.DataFrame(records)

    def summary_by_celltype(self) -> pd.DataFrame:
        """Summarize validation by cell type."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return df.groupby("cell_type").agg({
            "correlation": ["mean", "median", "count"],
            "r2": "mean",
            "pvalue": lambda x: (x < 0.05).sum(),
        }).round(3)

    def summary_by_cytokine(self) -> pd.DataFrame:
        """Summarize validation by cytokine."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return df.groupby("cytokine").agg({
            "correlation": ["mean", "median", "count"],
            "r2": "mean",
            "pvalue": lambda x: (x < 0.05).sum(),
        }).round(3)


class LinCytoSigValidator:
    """
    Validates LinCytoSig cell-type-specific cytokine signatures.

    For each signature like "Macrophage__IFNG":
    1. Filters to samples/cells of the matching cell type
    2. Correlates cytokine expression with inferred activity
    3. Optionally compares activity rank across cell types

    Example:
        >>> validator = LinCytoSigValidator()
        >>> summary = validator.validate(
        ...     activity=activity_df,
        ...     expression=expression_df,
        ...     metadata=metadata_df,
        ...     cell_type_col="cell_type"
        ... )
        >>> print(summary.mean_correlation)
        >>> summary.to_dataframe().to_csv("lincytosig_validation.csv")
    """

    # CytoSig name to HGNC symbol mapping
    CYTOSIG_TO_HGNC = {
        "TNFA": "TNF",
        "IFNA": "IFNA1",  # or could be IFNA2
        "IFNB": "IFNB1",
        "IFNG": "IFNG",
        "IFNL": "IFNL1",
        "TGFB1": "TGFB1",
        "TGFB3": "TGFB3",
        "IL1A": "IL1A",
        "IL1B": "IL1B",
        "IL2": "IL2",
        "IL3": "IL3",
        "IL4": "IL4",
        "IL6": "IL6",
        "IL10": "IL10",
        "IL12": "IL12A",
        "IL13": "IL13",
        "IL15": "IL15",
        "IL17A": "IL17A",
        "IL22": "IL22",
        "IL27": "IL27",
        "IL36A": "IL36A",
        "IL36B": "IL36B",
        "IL36G": "IL36G",
        "GMCSF": "CSF2",
        "GCSF": "CSF3",
        "MCSF": "CSF1",
        "EGF": "EGF",
        "HGF": "HGF",
        "FGF2": "FGF2",
        "FGF10": "FGF10",
        "VEGFA": "VEGFA",
        "VEGFC": "VEGFC",
        "PDGFB": "PDGFB",
        "PDGFD": "PDGFD",
        "BMP2": "BMP2",
        "BMP4": "BMP4",
        "BMP6": "BMP6",
        "BDNF": "BDNF",
        "GDNF": "GDNF",
        "NTF4": "NTF4",
        "NGF": "NGF",
        "NRG1": "NRG1",
        "IGF1": "IGF1",
        "INS": "INS",
        "GDF11": "GDF11",
        "MSTN": "MSTN",
        "Activin A": "INHBA",
        "OSM": "OSM",
        "LTA": "LTA",
        "TWEAK": "TNFSF12",
        "CD40L": "CD40LG",
        "HMGB1": "HMGB1",
        "ADM": "ADM",
        "CXCL12": "CXCL12",
        "TGFA": "TGFA",
        "WNT3A": "WNT3A",
        "NO": None,  # Not a gene
        "PDL1": "CD274",
    }

    def __init__(
        self,
        min_samples: int = 10,
        signature_sep: str = "__",
    ):
        """
        Initialize validator.

        Args:
            min_samples: Minimum samples required per cell type.
            signature_sep: Separator in signature names (default "__").
        """
        self.min_samples = min_samples
        self.signature_sep = signature_sep

    def parse_signature(self, signature: str) -> tuple[str, str]:
        """
        Parse LinCytoSig signature name into cell type and cytokine.

        Args:
            signature: Signature name (e.g., "Macrophage__IFNG").

        Returns:
            Tuple of (cell_type, cytokine).
        """
        parts = signature.split(self.signature_sep)
        if len(parts) != 2:
            raise ValueError(f"Invalid LinCytoSig signature format: {signature}")
        return parts[0], parts[1]

    def get_hgnc_symbol(self, cytokine: str) -> Optional[str]:
        """
        Get HGNC gene symbol for a CytoSig cytokine name.

        Args:
            cytokine: CytoSig cytokine name.

        Returns:
            HGNC symbol or None if not mappable.
        """
        return self.CYTOSIG_TO_HGNC.get(cytokine, cytokine)

    def validate(
        self,
        activity: pd.DataFrame,
        expression: pd.DataFrame,
        metadata: pd.DataFrame,
        cell_type_col: str = "cell_type",
        sample_col: Optional[str] = None,
        level: Literal["sample", "celltype"] = "sample",
    ) -> LinCytoSigValidationSummary:
        """
        Validate LinCytoSig signatures.

        Args:
            activity: Activity matrix (signatures x samples).
            expression: Expression matrix (genes x samples), z-scored recommended.
            metadata: Sample metadata with cell type information.
            cell_type_col: Column name for cell type.
            sample_col: Column for sample grouping (for pseudobulk).
            level: Validation level - "sample" or "celltype" aggregated.

        Returns:
            LinCytoSigValidationSummary with validation results.
        """
        # Align samples
        common_samples = list(
            set(activity.columns) & set(expression.columns) & set(metadata.index)
        )
        if len(common_samples) < self.min_samples:
            raise ValueError(f"Insufficient common samples: {len(common_samples)}")

        activity = activity[common_samples]
        expression = expression[common_samples]
        metadata = metadata.loc[common_samples]

        # Build gene name lookup (case-insensitive)
        expr_genes_upper = {g.upper(): g for g in expression.index}

        # Get all cell types and compute global rankings
        all_cell_types = metadata[cell_type_col].unique()

        results = []

        for signature in activity.index:
            # Parse signature
            try:
                cell_type, cytokine = self.parse_signature(signature)
            except ValueError:
                continue

            # Get HGNC symbol
            hgnc = self.get_hgnc_symbol(cytokine)
            if hgnc is None:
                continue

            # Find gene in expression matrix
            if hgnc.upper() not in expr_genes_upper:
                # Try original cytokine name
                if cytokine.upper() not in expr_genes_upper:
                    continue
                actual_gene = expr_genes_upper[cytokine.upper()]
            else:
                actual_gene = expr_genes_upper[hgnc.upper()]

            # Filter to matching cell type
            ct_mask = metadata[cell_type_col] == cell_type
            ct_samples = metadata.index[ct_mask].tolist()

            if len(ct_samples) < self.min_samples:
                continue

            # Get activity and expression for this cell type
            act_vals = activity.loc[signature, ct_samples].values
            expr_vals = expression.loc[actual_gene, ct_samples].values

            # Remove NaN
            mask = ~(np.isnan(act_vals) | np.isnan(expr_vals))
            if mask.sum() < self.min_samples:
                continue

            act_vals = act_vals[mask]
            expr_vals = expr_vals[mask]

            # Compute correlations
            r_pearson, p_pearson = stats.pearsonr(expr_vals, act_vals)
            r_spearman, _ = stats.spearmanr(expr_vals, act_vals)

            # Compute activity rank across cell types
            activity_rank = None
            expression_rank = None

            if level == "celltype" or len(all_cell_types) > 1:
                ct_activities = {}
                ct_expressions = {}

                for ct in all_cell_types:
                    ct_mask_all = metadata[cell_type_col] == ct
                    ct_samp = metadata.index[ct_mask_all].tolist()
                    if len(ct_samp) >= 3:
                        ct_activities[ct] = activity.loc[signature, ct_samp].mean()
                        ct_expressions[ct] = expression.loc[actual_gene, ct_samp].mean()

                if cell_type in ct_activities:
                    # Rank (1 = highest)
                    sorted_act = sorted(ct_activities.values(), reverse=True)
                    sorted_expr = sorted(ct_expressions.values(), reverse=True)
                    activity_rank = sorted_act.index(ct_activities[cell_type]) + 1
                    expression_rank = sorted_expr.index(ct_expressions[cell_type]) + 1

            results.append(LinCytoSigResult(
                signature=signature,
                cell_type=cell_type,
                cytokine=cytokine,
                correlation=r_pearson,
                spearman=r_spearman,
                pvalue=p_pearson,
                r2=r_pearson ** 2,
                n_samples=len(act_vals),
                mean_activity=np.mean(act_vals),
                mean_expression=np.mean(expr_vals),
                activity_rank=activity_rank,
                expression_rank=expression_rank,
            ))

        return LinCytoSigValidationSummary(results=results)

    def validate_cross_celltype(
        self,
        activity: pd.DataFrame,
        expression: pd.DataFrame,
        metadata: pd.DataFrame,
        cell_type_col: str = "cell_type",
        cytokine: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Validate that cell-type-specific signatures show highest activity
        in their target cell type.

        For each cytokine, checks if the signature for cell type X
        has higher activity in cell type X than in other cell types.

        Args:
            activity: Activity matrix.
            expression: Expression matrix.
            metadata: Sample metadata.
            cell_type_col: Cell type column.
            cytokine: Specific cytokine to validate (or all if None).

        Returns:
            DataFrame with cross-cell-type validation results.
        """
        # Align
        common = list(
            set(activity.columns) & set(expression.columns) & set(metadata.index)
        )
        activity = activity[common]
        expression = expression[common]
        metadata = metadata.loc[common]

        cell_types = metadata[cell_type_col].unique()
        records = []

        # Get unique cytokines from signatures
        cytokines_in_data = set()
        for sig in activity.index:
            try:
                _, cyto = self.parse_signature(sig)
                cytokines_in_data.add(cyto)
            except ValueError:
                continue

        if cytokine:
            cytokines_in_data = {cytokine} & cytokines_in_data

        for cyto in cytokines_in_data:
            # Get all cell types with this cytokine signature
            ct_sigs = [
                sig for sig in activity.index
                if sig.endswith(f"{self.signature_sep}{cyto}")
            ]

            if len(ct_sigs) < 2:
                continue

            # Compute mean activity per cell type for each signature
            for sig in ct_sigs:
                target_ct, _ = self.parse_signature(sig)

                ct_activities = {}
                for ct in cell_types:
                    ct_mask = metadata[cell_type_col] == ct
                    ct_samples = metadata.index[ct_mask].tolist()
                    if len(ct_samples) >= 3:
                        ct_activities[ct] = activity.loc[sig, ct_samples].mean()

                if target_ct not in ct_activities:
                    continue

                # Check if target cell type has highest activity
                target_activity = ct_activities[target_ct]
                all_activities = list(ct_activities.values())
                rank = sorted(all_activities, reverse=True).index(target_activity) + 1
                is_top = rank == 1
                percentile = (len(all_activities) - rank + 1) / len(all_activities) * 100

                records.append({
                    "signature": sig,
                    "cytokine": cyto,
                    "target_celltype": target_ct,
                    "target_activity": target_activity,
                    "rank": rank,
                    "n_celltypes": len(ct_activities),
                    "is_top": is_top,
                    "percentile": percentile,
                    "max_activity": max(all_activities),
                    "max_celltype": max(ct_activities, key=ct_activities.get),
                })

        return pd.DataFrame(records)


def validate_lincytosig(
    activity: pd.DataFrame,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
    min_samples: int = 10,
) -> LinCytoSigValidationSummary:
    """
    Convenience function for LinCytoSig validation.

    Args:
        activity: Activity matrix (signatures x samples).
        expression: Expression matrix (genes x samples).
        metadata: Sample metadata.
        cell_type_col: Cell type column name.
        min_samples: Minimum samples per cell type.

    Returns:
        LinCytoSigValidationSummary.
    """
    validator = LinCytoSigValidator(min_samples=min_samples)
    return validator.validate(activity, expression, metadata, cell_type_col)


def validate_lincytosig_specificity(
    activity: pd.DataFrame,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    """
    Validate that LinCytoSig signatures are cell-type-specific.

    Checks if each signature shows highest activity in its target cell type.

    Args:
        activity: Activity matrix.
        expression: Expression matrix.
        metadata: Sample metadata.
        cell_type_col: Cell type column.

    Returns:
        DataFrame with specificity results.
    """
    validator = LinCytoSigValidator()
    return validator.validate_cross_celltype(
        activity, expression, metadata, cell_type_col
    )
