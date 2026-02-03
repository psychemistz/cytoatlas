"""
Differential analysis pipeline.

GPU-accelerated group comparison for activity signatures.
"""

from cytoatlas_pipeline.differential.wilcoxon import (
    WilcoxonTest,
    wilcoxon_test,
)
from cytoatlas_pipeline.differential.ttest import (
    TTest,
    ttest,
)
from cytoatlas_pipeline.differential.effect_size import (
    compute_activity_diff,
    compute_cohens_d,
    EffectSizeCalculator,
)
from cytoatlas_pipeline.differential.fdr import (
    apply_fdr,
    FDRCorrector,
)
from cytoatlas_pipeline.differential.stratified import (
    StratifiedDifferential,
    DifferentialResult,
    run_differential,
)

__all__ = [
    # Wilcoxon
    "WilcoxonTest",
    "wilcoxon_test",
    # T-test
    "TTest",
    "ttest",
    # Effect size
    "compute_activity_diff",
    "compute_cohens_d",
    "EffectSizeCalculator",
    # FDR
    "apply_fdr",
    "FDRCorrector",
    # Stratified
    "StratifiedDifferential",
    "DifferentialResult",
    "run_differential",
]
