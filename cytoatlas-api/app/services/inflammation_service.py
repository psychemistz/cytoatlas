"""Inflammation Atlas data service."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.inflammation import (
    InflammationAgeBMIBoxplot,
    InflammationCellTypeActivity,
    InflammationCellTypeStratified,
    InflammationCohortValidation,
    InflammationCohortValidationResponse,
    InflammationCohortValidationSignature,
    InflammationCohortValidationSummary,
    InflammationConservedProgram,
    InflammationCorrelation,
    InflammationDifferential,
    InflammationDiseaseActivity,
    InflammationDiseaseComparison,
    InflammationDrivingPopulation,
    InflammationFeatureImportance,
    InflammationLongitudinal,
    InflammationROCCurve,
    InflammationSankeyData,
    InflammationSeverity,
    InflammationSummaryStats,
    InflammationTemporalResponse,
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
    async def get_disease_activity_summary(
        self,
        signature_type: str = "CytoSig",
    ) -> dict:
        """
        Get pre-aggregated disease activity data for fast visualization.

        Returns pre-computed aggregations instead of raw data to reduce
        data transfer from ~9MB to ~100KB.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Dictionary with:
            - heatmap: {z, x (signatures), y (diseases)} for disease × signature matrix
            - bar_data: {signature -> {cell_type -> mean_activity}} for bar charts
            - diseases: list of unique diseases
            - disease_groups: list of unique disease groups
            - signatures: list of unique signatures
            - cell_types: list of unique cell types
            - disease_to_group: mapping of disease to disease_group
        """
        data = await self.load_json("inflammation_disease.json")

        # Filter by signature type
        filtered = self.filter_by_signature_type(data, signature_type)

        if not filtered:
            return {
                "heatmap": {"z": [], "x": [], "y": []},
                "bar_data": {},
                "diseases": [],
                "disease_groups": [],
                "signatures": [],
                "cell_types": [],
                "disease_to_group": {},
            }

        # Extract unique values
        diseases = sorted(set(r.get("disease") for r in filtered if r.get("disease")))
        disease_groups = sorted(set(r.get("disease_group") for r in filtered if r.get("disease_group")))
        signatures = sorted(set(r.get("signature") for r in filtered if r.get("signature")))
        cell_types = sorted(set(r.get("cell_type") for r in filtered if r.get("cell_type")))

        # Build disease to group mapping
        disease_to_group = {}
        for r in filtered:
            if r.get("disease") and r.get("disease_group"):
                disease_to_group[r["disease"]] = r["disease_group"]

        # Pre-aggregate for heatmap: disease × signature (mean across cell types)
        disease_sig_activity = {}
        for r in filtered:
            key = (r.get("disease"), r.get("signature"))
            if key[0] and key[1]:
                if key not in disease_sig_activity:
                    disease_sig_activity[key] = []
                disease_sig_activity[key].append(r.get("mean_activity", 0))

        # Build heatmap matrix
        z_data = []
        for disease in diseases:
            row = []
            for sig in signatures:
                vals = disease_sig_activity.get((disease, sig), [0])
                row.append(sum(vals) / len(vals) if vals else 0)
            z_data.append(row)

        # Pre-aggregate for bar charts: signature -> cell_type -> mean_activity
        # Group by (signature, cell_type, disease_group)
        sig_ct_activity = {}
        for r in filtered:
            sig = r.get("signature")
            ct = r.get("cell_type")
            dg = r.get("disease_group", "all")
            if sig and ct:
                key = (sig, ct, dg)
                if key not in sig_ct_activity:
                    sig_ct_activity[key] = []
                sig_ct_activity[key].append(r.get("mean_activity", 0))

        # Build bar_data structure: {signature: {disease_group: [{cell_type, mean_activity}, ...]}}
        bar_data = {}
        for sig in signatures:
            bar_data[sig] = {"all": {}}
            for dg in disease_groups:
                bar_data[sig][dg] = {}

        for (sig, ct, dg), vals in sig_ct_activity.items():
            mean_val = sum(vals) / len(vals) if vals else 0
            # Add to disease group
            if dg in bar_data.get(sig, {}):
                bar_data[sig][dg][ct] = mean_val
            # Add to "all" aggregation
            if ct not in bar_data[sig]["all"]:
                bar_data[sig]["all"][ct] = []
            bar_data[sig]["all"][ct].append(mean_val)

        # Finalize "all" aggregation (mean across disease groups)
        for sig in signatures:
            for ct, vals in bar_data[sig]["all"].items():
                if isinstance(vals, list):
                    bar_data[sig]["all"][ct] = sum(vals) / len(vals) if vals else 0

        return {
            "heatmap": {"z": z_data, "x": signatures, "y": diseases},
            "bar_data": bar_data,
            "diseases": diseases,
            "disease_groups": disease_groups,
            "signatures": signatures,
            "cell_types": cell_types,
            "disease_to_group": disease_to_group,
        }

    @cached(prefix="inflammation", ttl=3600)
    async def get_differential(
        self,
        disease: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[InflammationDifferential]:
        """
        Get disease vs healthy differential analysis.

        This loads from pre-computed differential data (not cell-type stratified).

        Args:
            disease: Optional disease filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of differential results with log2FC, p-value, etc.
        """
        data = await self.load_json("inflammation_differential.json")

        results = []
        for item in data:
            # Map 'signature' field to 'signature_type' (data format quirk)
            sig_type = item.get("signature", "CytoSig")

            # Apply signature type filter
            if signature_type and sig_type != signature_type:
                continue

            # Apply disease filter
            if disease and disease != "all" and item.get("disease") != disease:
                continue

            results.append(
                InflammationDifferential(
                    signature=item.get("protein", ""),
                    signature_type=sig_type,
                    disease=item.get("disease", ""),
                    group1=item.get("group1", ""),
                    group2=item.get("group2", "healthy"),
                    mean_g1=item.get("mean_g1", 0),
                    mean_g2=item.get("mean_g2", 0),
                    n_g1=item.get("n_g1", 0),
                    n_g2=item.get("n_g2", 0),
                    activity_diff=item.get("activity_diff", 0),
                    pvalue=item.get("pvalue", 1.0),
                    qvalue=item.get("qvalue"),
                    neg_log10_pval=item.get("neg_log10_pval"),
                )
            )

        return results

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
    ) -> InflammationCohortValidationResponse:
        """
        Get cross-cohort validation results.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Cohort validation response with correlations and summary
        """
        data = await self.load_json("cohort_validation.json")

        # Extract correlations array and filter by signature type
        correlations_raw = data.get("correlations", [])
        correlations_filtered = self.filter_by_signature_type(
            correlations_raw, signature_type
        )
        correlations = [
            InflammationCohortValidationSignature(**r) for r in correlations_filtered
        ]

        # Extract consistency array and filter by signature type
        consistency_raw = data.get("consistency", [])
        consistency_filtered = self.filter_by_signature_type(
            consistency_raw, signature_type
        )
        consistency = [
            InflammationCohortValidationSummary(**r) for r in consistency_filtered
        ]

        return InflammationCohortValidationResponse(
            correlations=correlations,
            consistency=consistency,
        )

    @cached(prefix="inflammation", ttl=3600)
    async def get_disease_sankey(self) -> InflammationSankeyData:
        """
        Get Sankey diagram data for disease flow.

        Returns:
            Sankey data with nodes and links for visualization
        """
        data = await self.load_json("disease_sankey.json")

        return InflammationSankeyData(**data)

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
        # Load from pre-computed cell drivers data
        data = await self.load_json("inflammation_cell_drivers.json")
        effects = data.get("effects", [])

        results = []
        for item in effects:
            # Apply filters
            if signature_type and item.get("signature_type") != signature_type:
                continue
            if disease and item.get("disease") != disease:
                continue

            activity_diff = item.get("effect", 0)
            pvalue = item.get("pvalue", 1.0)
            # Consider significant and large effect as "driving"
            is_driving = pvalue < 0.05 and abs(activity_diff) > 0.5

            results.append(
                InflammationCellTypeStratified(
                    cell_type=item.get("cell_type", ""),
                    signature=item.get("signature", item.get("cytokine", "")),
                    signature_type=item.get("signature_type", ""),
                    disease=item.get("disease", ""),
                    activity_diff=activity_diff,
                    p_value=pvalue,
                    q_value=None,
                    is_driving=is_driving,
                )
            )

        return results

    @cached(prefix="inflammation", ttl=3600)
    async def get_cell_drivers_raw(self) -> dict:
        """
        Get raw cell drivers data for direct frontend use.

        Returns the full inflammation_cell_drivers.json structure unchanged,
        matching the format expected by index.html.
        """
        return await self.load_json("inflammation_cell_drivers.json")

    @cached(prefix="inflammation", ttl=3600)
    async def get_driving_populations(
        self,
        disease: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[InflammationDrivingPopulation]:
        """
        Get driving cell populations for diseases.

        Args:
            disease: Optional disease filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of driving population results
        """
        # Compute from stratified analysis
        stratified = await self.get_celltype_stratified(disease=disease, signature_type=signature_type)

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
                    "activity_diff_sum": 0,
                }
            driving_map[key]["signatures"].append(item.signature)
            driving_map[key]["activity_diff_sum"] += abs(item.activity_diff)

        results = []
        for key, data in driving_map.items():
            results.append(
                InflammationDrivingPopulation(
                    disease=data["disease"],
                    cell_type=data["cell_type"],
                    n_signatures=len(data["signatures"]),
                    top_signatures=data["signatures"][:5],
                    mean_activity_diff=data["activity_diff_sum"] / len(data["signatures"]),
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
                        mean_activity_diff=data["activity_sum"] / len(data["diseases"]),
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

    @cached(prefix="inflammation", ttl=3600)
    async def get_severity_analysis(
        self,
        disease: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[InflammationSeverity]:
        """
        Get disease severity correlation analysis.

        Args:
            disease: Optional disease filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of severity analysis results
        """
        data = await self.load_json("inflammation_severity.json")

        results = self.filter_by_signature_type(data, signature_type)

        if disease:
            results = [r for r in results if r.get("disease") == disease]

        # Sort by disease, then severity order, then signature
        results.sort(key=lambda x: (x.get("disease", ""), x.get("severity_order", 99), x.get("signature", "")))

        return [InflammationSeverity(**r) for r in results]

    @cached(prefix="inflammation", ttl=3600)
    async def get_severity_raw(self) -> list[dict]:
        """
        Get raw severity data for direct frontend use.

        Returns the full inflammation_severity.json array unchanged,
        matching the format expected by index.html.
        """
        return await self.load_json("inflammation_severity.json")

    @cached(prefix="inflammation", ttl=3600)
    async def get_differential_raw(self) -> list[dict]:
        """
        Get raw differential data for direct frontend use.

        Returns the full inflammation_differential.json array unchanged,
        matching the format expected by index.html.

        Fields: protein, disease, signature (CytoSig/SecAct), comparison,
        healthy_note, n_g1, n_g2, activity_diff, pvalue, qvalue, neg_log10_pval
        """
        return await self.load_json("inflammation_differential.json")

    @cached(prefix="inflammation", ttl=3600)
    async def get_treatment_response_raw(self) -> dict:
        """
        Get raw treatment response data for direct frontend use.

        Returns the full treatment_response.json structure unchanged,
        matching the format expected by index.html.

        Structure: {
            roc_curves: [...],  // disease, model, signature_type, auc, fpr, tpr
            feature_importance: [...],  // disease, model, signature_type, feature, importance
            predictions: [...]  // disease, signature_type, response, probability, model
        }
        """
        return await self.load_json("treatment_response.json")

    async def get_severity_diseases(self) -> list[str]:
        """Get list of diseases with severity data."""
        data = await self.load_json("inflammation_severity.json")
        diseases = sorted(set(r.get("disease") for r in data if r.get("disease")))
        return diseases

    async def get_severity_levels(self, disease: str) -> list[str]:
        """Get severity levels for a specific disease, ordered from mild to severe."""
        data = await self.load_json("inflammation_severity.json")
        disease_data = [r for r in data if r.get("disease") == disease]

        # Get unique severities with their order
        severity_order = {}
        for r in disease_data:
            sev = r.get("severity")
            order = r.get("severity_order", 99)
            if sev and sev not in severity_order:
                severity_order[sev] = order

        # Sort by order
        return sorted(severity_order.keys(), key=lambda x: severity_order.get(x, 99))

    @cached(prefix="inflammation", ttl=3600)
    async def get_temporal_analysis(self) -> InflammationTemporalResponse:
        """
        Get temporal/longitudinal analysis data.

        Note: The Inflammation Atlas is cross-sectional, so this shows
        comparison between sampling timepoints, not the same patients over time.

        Returns:
            Temporal analysis response with distribution and activity data
        """
        data = await self.load_json("inflammation_longitudinal.json")

        # Convert activity records to Pydantic models
        activity_records = [
            InflammationLongitudinal(**r) for r in data.get("timepoint_activity", [])
        ]

        return InflammationTemporalResponse(
            has_longitudinal=data.get("has_longitudinal", False),
            note=data.get("note", ""),
            timepoint_distribution=data.get("timepoint_distribution", {}),
            disease_timepoints=data.get("disease_timepoints", {}),
            timepoint_activity=activity_records,
            treatment_by_timepoint=data.get("treatment_by_timepoint", {}),
        )

    @cached(prefix="inflammation", ttl=3600)
    async def get_temporal_activity(
        self,
        disease: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[InflammationLongitudinal]:
        """
        Get activity data by timepoint.

        Args:
            disease: Optional disease filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of temporal activity records
        """
        data = await self.load_json("inflammation_longitudinal.json")

        results = data.get("timepoint_activity", [])
        results = self.filter_by_signature_type(results, signature_type)

        if disease:
            results = [r for r in results if r.get("disease") == disease]

        # Sort by disease, timepoint, signature
        results.sort(key=lambda x: (x.get("disease", ""), x.get("timepoint_num", 0), x.get("signature", "")))

        return [InflammationLongitudinal(**r) for r in results]

    async def get_temporal_diseases(self) -> list[str]:
        """Get list of diseases with temporal (multi-timepoint) data."""
        data = await self.load_json("inflammation_longitudinal.json")
        activity = data.get("timepoint_activity", [])

        # Get diseases that have activity at multiple timepoints
        disease_timepoints = {}
        for r in activity:
            disease = r.get("disease")
            tp = r.get("timepoint")
            if disease not in disease_timepoints:
                disease_timepoints[disease] = set()
            disease_timepoints[disease].add(tp)

        # Return diseases with >1 timepoint
        return sorted([d for d, tps in disease_timepoints.items() if len(tps) > 1])
