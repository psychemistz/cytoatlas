"""Inflammation Atlas data service."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.inflammation import (
    InflammationAgeBMIBoxplot,
    InflammationCellTypeActivity,
    InflammationCellTypeStratified,
    InflammationCohortValidation,
    InflammationConservedProgram,
    InflammationCorrelation,
    InflammationDiseaseActivity,
    InflammationDiseaseComparison,
    InflammationDrivingPopulation,
    InflammationFeatureImportance,
    InflammationROCCurve,
    InflammationSankeyData,
    InflammationSummaryStats,
    InflammationTreatmentResponse,
)
from app.services.base import BaseService

settings = get_settings()


class InflammationService(BaseService):
    """Service for Inflammation Atlas data."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path
        self.results_dir = settings.inflammation_results_dir

    @cached(prefix="inflammation", ttl=3600)
    async def get_cell_type_activity(
        self,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> list[InflammationCellTypeActivity]:
        """
        Get mean activity by cell type.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            List of cell type activity results
        """
        data = await self.load_json("inflammation_celltype.json")

        results = self.filter_by_signature_type(data, signature_type)

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return [InflammationCellTypeActivity(**r) for r in results]

    @cached(prefix="inflammation", ttl=3600)
    async def get_disease_activity(
        self,
        disease: str | None = None,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> list[InflammationDiseaseActivity]:
        """
        Get disease-specific activity data.

        Args:
            disease: Optional disease filter
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            List of disease activity results
        """
        data = await self.load_json("inflammation_disease.json")

        results = self.filter_by_signature_type(data, signature_type)

        if disease:
            results = [r for r in results if r.get("disease") == disease]

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return [InflammationDiseaseActivity(**r) for r in results]

    # Alias for backward compatibility
    async def get_disease_comparison(
        self,
        disease: str | None = None,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> list[InflammationDiseaseActivity]:
        """Alias for get_disease_activity (backward compatibility)."""
        return await self.get_disease_activity(disease, signature_type, cell_type)

    @cached(prefix="inflammation", ttl=3600)
    async def get_treatment_response_summary(
        self,
        disease: str | None = None,
    ) -> list[InflammationTreatmentResponse]:
        """
        Get treatment response prediction summary.

        Args:
            disease: Optional disease filter

        Returns:
            List of treatment response summaries
        """
        data = await self.load_json("treatment_response.json")

        # Extract summary from roc_curves
        roc_curves = data.get("roc_curves", [])

        results = []
        for curve in roc_curves:
            if disease and curve.get("disease") != disease:
                continue

            results.append(
                InflammationTreatmentResponse(
                    disease=curve["disease"],
                    model=curve["model"],
                    signature_type="CytoSig",  # Treatment uses CytoSig
                    auc=curve["auc"],
                    n_samples=curve["n_samples"],
                    n_responders=0,  # Would need to compute from predictions
                    n_non_responders=0,
                )
            )

        return results

    @cached(prefix="inflammation", ttl=3600)
    async def get_roc_curves(
        self,
        disease: str | None = None,
        model: str | None = None,
    ) -> list[InflammationROCCurve]:
        """
        Get ROC curve data for treatment response.

        Args:
            disease: Optional disease filter
            model: Optional model filter ('Logistic Regression', 'Random Forest')

        Returns:
            List of ROC curve data
        """
        data = await self.load_json("treatment_response.json")

        roc_curves = data.get("roc_curves", [])

        results = []
        for curve in roc_curves:
            if disease and curve.get("disease") != disease:
                continue
            if model and curve.get("model") != model:
                continue

            results.append(InflammationROCCurve(**curve))

        return results

    @cached(prefix="inflammation", ttl=3600)
    async def get_feature_importance(
        self,
        disease: str | None = None,
        model: str = "Random Forest",
    ) -> list[InflammationFeatureImportance]:
        """
        Get feature importance for treatment response model.

        Args:
            disease: Optional disease filter
            model: Model type

        Returns:
            List of feature importance results
        """
        data = await self.load_json("treatment_response.json")

        importance = data.get("feature_importance", [])

        results = []
        for item in importance:
            if disease and item.get("disease") != disease:
                continue
            if item.get("model") != model:
                continue

            results.append(InflammationFeatureImportance(**item))

        # Sort by importance and add rank
        results.sort(key=lambda x: x.importance, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    @cached(prefix="inflammation", ttl=3600)
    async def get_cohort_validation(
        self,
        signature_type: str = "CytoSig",
    ) -> list[InflammationCohortValidation]:
        """
        Get cross-cohort validation results.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of cohort validation results
        """
        data = await self.load_json("cohort_validation.json")

        results = self.filter_by_signature_type(data, signature_type)

        return [InflammationCohortValidation(**r) for r in results]

    @cached(prefix="inflammation", ttl=3600)
    async def get_disease_sankey(self) -> list[InflammationSankeyData]:
        """
        Get Sankey diagram data for disease flow.

        Returns:
            List of Sankey node/link data
        """
        data = await self.load_json("disease_sankey.json")

        return [InflammationSankeyData(**item) for item in data]

    @cached(prefix="inflammation", ttl=3600)
    async def get_correlations(
        self,
        variable: str,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> list[InflammationCorrelation]:
        """
        Get correlation data (age, BMI).

        Args:
            variable: 'age' or 'bmi'
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            List of correlation results
        """
        # Use cell-type specific correlations
        data = await self.load_json("inflammation_celltype_correlations.json")

        if variable not in data:
            raise ValueError(f"Invalid variable: {variable}")

        results = data[variable]

        # Transform field names: protein -> signature
        # Note: Original data may have "protein" as the signature name
        transformed = []
        for r in results:
            record = {
                "cell_type": r.get("cell_type", "All"),
                "signature": r.get("protein", r.get("signature")),
                "signature_type": signature_type,  # Use the parameter value
                "variable": variable,
                "rho": r.get("rho", 0),
                "p_value": r.get("pvalue", r.get("p_value", 1)),  # Accept both field names
                "q_value": r.get("qvalue", r.get("q_value")),
                "n_samples": r.get("n"),
            }
            transformed.append(record)

        # Filter by signature type
        results = self.filter_by_signature_type(transformed, signature_type)

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return [InflammationCorrelation(**r) for r in results]

    @cached(prefix="inflammation", ttl=3600)
    async def get_cell_type_correlations(
        self,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get cell type correlation matrix.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Cell type correlation data
        """
        data = await self.load_json("inflammation_celltype_correlations.json")

        results = self.filter_by_signature_type(data, signature_type)

        return results

    @cached(prefix="inflammation", ttl=3600)
    async def get_celltype_stratified(
        self,
        disease: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[InflammationCellTypeStratified]:
        """
        Get cell type stratified disease analysis.

        Args:
            disease: Optional disease filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of stratified results
        """
        # Derive from disease activity data
        data = await self.get_disease_activity(
            disease=disease, signature_type=signature_type
        )

        results = []
        for item in data:
            # Use mean_activity as proxy for log2fc since actual comparison data not available
            log2fc = item.mean_activity
            # Consider high activity as "driving"
            is_driving = abs(log2fc) > 1

            results.append(
                InflammationCellTypeStratified(
                    cell_type=item.cell_type,
                    signature=item.signature,
                    signature_type=item.signature_type,
                    disease=item.disease,
                    log2fc=log2fc,
                    p_value=0.01,  # Placeholder - actual p-values not in this data
                    q_value=None,
                    is_driving=is_driving,
                )
            )

        return results

    @cached(prefix="inflammation", ttl=3600)
    async def get_driving_populations(
        self,
        disease: str | None = None,
    ) -> list[InflammationDrivingPopulation]:
        """
        Get driving cell populations for diseases.

        Args:
            disease: Optional disease filter

        Returns:
            List of driving population results
        """
        # Compute from stratified analysis
        stratified = await self.get_celltype_stratified(disease=disease)

        # Group by disease and cell type
        driving_map: dict = {}
        for item in stratified:
            if not item.is_driving:
                continue

            key = (item.disease, item.cell_type)
            if key not in driving_map:
                driving_map[key] = {
                    "disease": item.disease,
                    "cell_type": item.cell_type,
                    "signatures": [],
                    "log2fc_sum": 0,
                }
            driving_map[key]["signatures"].append(item.signature)
            driving_map[key]["log2fc_sum"] += abs(item.log2fc)

        results = []
        for key, data in driving_map.items():
            results.append(
                InflammationDrivingPopulation(
                    disease=data["disease"],
                    cell_type=data["cell_type"],
                    n_signatures=len(data["signatures"]),
                    top_signatures=data["signatures"][:5],
                    mean_log2fc=data["log2fc_sum"] / len(data["signatures"]),
                )
            )

        return sorted(results, key=lambda x: x.n_signatures, reverse=True)

    @cached(prefix="inflammation", ttl=3600)
    async def get_conserved_programs(
        self,
        signature_type: str = "CytoSig",
        min_diseases: int = 3,
    ) -> list[InflammationConservedProgram]:
        """
        Get conserved cytokine programs across diseases.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            min_diseases: Minimum diseases for conservation

        Returns:
            List of conserved programs
        """
        # Get disease activity data
        disease_data = await self.get_disease_activity(signature_type=signature_type)

        # Group by signature
        sig_diseases: dict = {}
        for item in disease_data:
            if item.signature not in sig_diseases:
                sig_diseases[item.signature] = {
                    "diseases": set(),
                    "activity_sum": 0,
                    "directions": [],
                }

            # Consider high activity signatures
            if abs(item.mean_activity) > 0.5:
                sig_diseases[item.signature]["diseases"].add(item.disease)
                sig_diseases[item.signature]["activity_sum"] += item.mean_activity
                sig_diseases[item.signature]["directions"].append(
                    1 if item.mean_activity > 0 else -1
                )

        results = []
        for sig, data in sig_diseases.items():
            if len(data["diseases"]) >= min_diseases:
                directions = data["directions"]
                consistency = (
                    abs(sum(directions)) / len(directions) if directions else 0
                )

                results.append(
                    InflammationConservedProgram(
                        signature=sig,
                        signature_type=signature_type,
                        diseases=sorted(list(data["diseases"])),
                        n_diseases=len(data["diseases"]),
                        mean_log2fc=data["activity_sum"] / len(data["diseases"]),
                        consistency_score=consistency,
                    )
                )

        return sorted(results, key=lambda x: x.n_diseases, reverse=True)

    async def get_summary_stats(self) -> InflammationSummaryStats:
        """Get Inflammation Atlas summary statistics."""
        data = await self.load_json("summary_stats.json")

        inflam_stats = data.get("inflammation", {})

        # Get unique diseases from disease activity data
        disease_data = await self.get_disease_activity()
        diseases = sorted(list(set(d.disease for d in disease_data)))
        disease_groups = sorted(list(set(d.disease_group for d in disease_data)))

        # Get cell types
        cell_type_data = await self.get_cell_type_activity()
        cell_types = sorted(list(set(c.cell_type for c in cell_type_data)))

        return InflammationSummaryStats(
            n_samples=inflam_stats.get("n_samples", 0),
            n_cell_types=len(cell_types),
            n_cells=inflam_stats.get("n_cells", 0),
            n_diseases=len(diseases),
            n_disease_groups=len(disease_groups),
            diseases=diseases,
            disease_groups=disease_groups,
            cohorts=["main", "validation", "external"],
        )

    async def get_available_diseases(self) -> list[str]:
        """Get list of available diseases."""
        data = await self.get_disease_activity()
        return sorted(list(set(d.disease for d in data)))

    async def get_available_cell_types(self) -> list[str]:
        """Get list of available cell types."""
        data = await self.get_cell_type_activity()
        return sorted(list(set(c.cell_type for c in data)))

    @cached(prefix="inflammation", ttl=3600)
    async def get_age_bmi_boxplots(
        self,
        signature: str,
        signature_type: str = "CytoSig",
        stratify_by: str = "age",
        cell_type: str | None = None,
    ) -> list[InflammationAgeBMIBoxplot]:
        """
        Get pre-computed age/BMI boxplot data.

        Args:
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'
            stratify_by: 'age' or 'bmi'
            cell_type: Optional cell type filter

        Returns:
            Boxplot statistics per bin
        """
        data = await self.load_json("age_bmi_boxplots.json")

        # Data structure: {"cima": {...}, "inflammation": {"age": [...], "bmi": [...]}}
        inflam_data = data.get("inflammation", {})
        results = inflam_data.get(stratify_by, [])

        if not results:
            return []

        # Filter by signature type (data uses "sig_type")
        results = [r for r in results if r.get("sig_type") == signature_type]

        # Filter by signature
        results = [r for r in results if r.get("signature") == signature]

        # Filter by cell type
        if cell_type:
            # Return cell-type specific data
            results = [r for r in results if r.get("cell_type") == cell_type]
        else:
            # Return sample-level data only (cell_type is None or 'All')
            results = [r for r in results if r.get("cell_type") in (None, "All")]

        return [InflammationAgeBMIBoxplot(**r) for r in results]

    @cached(prefix="inflammation", ttl=3600)
    async def get_stratified_heatmap(
        self,
        signature: str,
        signature_type: str = "CytoSig",
        stratify_by: str = "age",
    ) -> dict:
        """
        Get cell type × bin heatmap data for a signature.

        Args:
            signature: Signature name
            signature_type: 'CytoSig' or 'SecAct'
            stratify_by: 'age' or 'bmi'

        Returns:
            Dict with cell_types, bins, and medians matrix
        """
        data = await self.load_json("age_bmi_boxplots.json")

        inflam_data = data.get("inflammation", {})
        results = inflam_data.get(stratify_by, [])

        if not results:
            return {"cell_types": [], "bins": [], "medians": []}

        # Filter by signature type and signature
        results = [r for r in results if r.get("sig_type") == signature_type]
        results = [r for r in results if r.get("signature") == signature]

        # Only get cell-type specific data (not sample-level)
        results = [r for r in results if r.get("cell_type") not in (None, "All")]

        if not results:
            return {"cell_types": [], "bins": [], "medians": []}

        # Get unique cell types and bins
        cell_types = sorted(list(set(r.get("cell_type") for r in results)))

        if stratify_by == "age":
            bins = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
        else:
            bins = ["Underweight", "Normal", "Overweight", "Obese"]

        # Build median matrix: cell_types × bins
        medians = []
        for ct in cell_types:
            row = []
            for bin_name in bins:
                matching = [r for r in results if r.get("cell_type") == ct and r.get("bin") == bin_name]
                if matching:
                    row.append(matching[0].get("median", 0))
                else:
                    row.append(None)
            medians.append(row)

        return {
            "cell_types": cell_types,
            "bins": bins,
            "medians": medians,
        }
