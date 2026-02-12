"""Cross-atlas comparison service."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.schemas.cross_atlas import (
    AtlasComparisonData,
    CrossAtlasCellTypeMapping,
    CrossAtlasComparison,
    CrossAtlasConservedSignature,
    CrossAtlasCorrelation,
    CrossAtlasMetaAnalysis,
    CrossAtlasPathwayEnrichment,
    CrossAtlasSummary,
)
from app.services.base import BaseService

settings = get_settings()


class CrossAtlasService(BaseService):
    """Service for cross-atlas comparison data."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_atlas_correlations(
        self,
        signature_type: str = "CytoSig",
        atlas1: str | None = None,
        atlas2: str | None = None,
    ) -> list[CrossAtlasCorrelation]:
        """
        Get cross-atlas signature correlations.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            atlas1: Optional first atlas filter
            atlas2: Optional second atlas filter

        Returns:
            List of correlation results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle different data formats
        correlations_raw = data.get("correlations", data.get("correlation", {}))

        results = []

        # If it's the new dict format: {cima_vs_inflammation: {correlation, pvalue, n}, ...}
        if isinstance(correlations_raw, dict) and "cima_vs_inflammation" in correlations_raw:
            atlas_pairs = [
                ("CIMA", "Inflammation", "cima_vs_inflammation"),
                ("CIMA", "scAtlas", "cima_vs_scatlas"),
                ("Inflammation", "scAtlas", "inflam_vs_scatlas"),
            ]
            for a1, a2, key in atlas_pairs:
                pair_data = correlations_raw.get(key, {})
                if pair_data:
                    results.append({
                        "signature": "Overall",
                        "signature_type": signature_type,
                        "atlas1": a1,
                        "atlas2": a2,
                        "correlation": pair_data.get("correlation", 0),
                        "p_value": pair_data.get("pvalue", 1),
                        "n_common_cell_types": pair_data.get("n", 0),
                    })
        # If it's already a list format
        elif isinstance(correlations_raw, list):
            results = self.filter_by_signature_type(correlations_raw, signature_type)

        if atlas1:
            results = [r for r in results if r.get("atlas1") == atlas1]

        if atlas2:
            results = [r for r in results if r.get("atlas2") == atlas2]

        return [CrossAtlasCorrelation(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_cell_type_mappings(self) -> list[CrossAtlasCellTypeMapping]:
        """
        Get cell type mappings between atlases.

        Returns:
            List of cell type mapping results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle both key naming conventions
        mappings_raw = data.get("cell_type_mappings", data.get("shared_cell_types", []))

        # If it's a list of strings (cell type names), convert to mapping objects
        if mappings_raw and isinstance(mappings_raw[0], str):
            mappings = []
            for ct_name in mappings_raw:
                mappings.append({
                    "cell_type_cima": ct_name,
                    "cell_type_inflammation": ct_name,
                    "cell_type_scatlas": ct_name,
                    "harmonized_name": ct_name,
                    "confidence": 1.0,
                })
            return [CrossAtlasCellTypeMapping(**m) for m in mappings]

        # If it's already the correct format
        return [CrossAtlasCellTypeMapping(**m) for m in mappings_raw]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_conserved_signatures(
        self,
        signature_type: str = "CytoSig",
        min_atlases: int = 2,
    ) -> list[CrossAtlasConservedSignature]:
        """
        Get conserved signatures across atlases.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            min_atlases: Minimum number of atlases

        Returns:
            List of conserved signature results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle both key naming conventions (conserved vs conserved_signatures)
        conserved_raw = data.get("conserved_signatures", data.get("conserved", {}))
        # Conserved may be dict with "signatures" key or list
        if isinstance(conserved_raw, dict):
            conserved = conserved_raw.get("signatures", [])
        else:
            conserved = conserved_raw

        # Transform data to match schema
        results = []
        for r in conserved:
            # Build atlases list from boolean flags
            atlases = []
            if r.get("cima"):
                atlases.append("CIMA")
            if r.get("inflammation"):
                atlases.append("Inflammation")
            if r.get("scatlas"):
                atlases.append("scAtlas")

            # Calculate mean and std from atlas-specific means
            means = []
            if r.get("cima_mean") is not None:
                means.append(r["cima_mean"])
            if r.get("inflammation_mean") is not None:
                means.append(r["inflammation_mean"])
            if r.get("scatlas_mean") is not None:
                means.append(r["scatlas_mean"])

            mean_activity = sum(means) / len(means) if means else 0
            std_activity = (
                (sum((m - mean_activity) ** 2 for m in means) / len(means)) ** 0.5
                if len(means) > 1
                else 0
            )

            # Consistency score: inverse of CV, or 1 if std is 0
            consistency = 1.0 if std_activity == 0 else 1.0 / (1 + abs(std_activity / mean_activity) if mean_activity != 0 else 1)

            results.append({
                "signature": r.get("signature", ""),
                "signature_type": signature_type,
                "atlases": atlases,
                "n_atlases": r.get("n_atlases", len(atlases)),
                "mean_activity": mean_activity,
                "std_activity": std_activity,
                "consistency_score": consistency,
                "top_cell_types": [],  # Not available in current data
            })

        results = [r for r in results if r.get("n_atlases", 0) >= min_atlases]

        return [CrossAtlasConservedSignature(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_meta_analysis(
        self,
        signature_type: str = "CytoSig",
        signature: str | None = None,
        cell_type: str | None = None,
    ) -> list[CrossAtlasMetaAnalysis]:
        """
        Get meta-analysis results across atlases.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            signature: Optional signature filter
            cell_type: Optional cell type filter

        Returns:
            List of meta-analysis results
        """
        data = await self.load_json("cross_atlas.json")

        meta = data.get("meta_analysis", [])
        results = self.filter_by_signature_type(meta, signature_type)

        if signature:
            results = [r for r in results if r.get("signature") == signature]

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return [CrossAtlasMetaAnalysis(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_pathway_enrichment(
        self,
        signature_type: str = "CytoSig",
        signature: str | None = None,
        pathway_database: str | None = None,
    ) -> list[CrossAtlasPathwayEnrichment]:
        """
        Get pathway enrichment across atlases.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            signature: Optional signature filter
            pathway_database: Optional database filter ('KEGG', 'GO', 'Reactome')

        Returns:
            List of pathway enrichment results
        """
        data = await self.load_json("cross_atlas.json")

        # Handle both key naming conventions
        enrichment = data.get("pathway_enrichment", data.get("pathways", []))
        results = self.filter_by_signature_type(enrichment, signature_type)

        if signature:
            results = [r for r in results if r.get("signature") == signature]

        if pathway_database:
            results = [r for r in results if r.get("pathway_database") == pathway_database]

        return [CrossAtlasPathwayEnrichment(**r) for r in results]

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_comparison_data(
        self,
        signature_type: str = "CytoSig",
        cell_type: str | None = None,
    ) -> AtlasComparisonData:
        """
        Get full atlas comparison dataset.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cell_type: Optional cell type filter

        Returns:
            Full comparison data with summary
        """
        data = await self.load_json("cross_atlas.json")

        # Handle different data formats
        comparisons_raw = data.get("comparisons", data.get("comparison", {}))

        comparisons = []

        # If it's the new dict format with atlas pairs
        if isinstance(comparisons_raw, dict) and "cima_vs_inflammation" in comparisons_raw:
            # Aggregate data from all atlas pairs
            # Build a lookup: (signature, cell_type) -> {cima, inflammation, scatlas}
            activity_lookup: dict = {}

            # CIMA vs Inflammation: x=CIMA, y=Inflammation
            cima_inflam = comparisons_raw.get("cima_vs_inflammation", {}).get("data", [])
            for point in cima_inflam:
                key = (point.get("signature"), point.get("cell_type"))
                if key not in activity_lookup:
                    activity_lookup[key] = {"cima": None, "inflammation": None, "scatlas": None}
                activity_lookup[key]["cima"] = point.get("x")
                activity_lookup[key]["inflammation"] = point.get("y")

            # CIMA vs scAtlas: x=CIMA, y=scAtlas
            cima_scatlas = comparisons_raw.get("cima_vs_scatlas", {}).get("data", [])
            for point in cima_scatlas:
                key = (point.get("signature"), point.get("cell_type"))
                if key not in activity_lookup:
                    activity_lookup[key] = {"cima": None, "inflammation": None, "scatlas": None}
                activity_lookup[key]["cima"] = point.get("x")
                activity_lookup[key]["scatlas"] = point.get("y")

            # Inflammation vs scAtlas: x=Inflammation, y=scAtlas
            inflam_scatlas = comparisons_raw.get("inflam_vs_scatlas", {}).get("data", [])
            for point in inflam_scatlas:
                key = (point.get("signature"), point.get("cell_type"))
                if key not in activity_lookup:
                    activity_lookup[key] = {"cima": None, "inflammation": None, "scatlas": None}
                activity_lookup[key]["inflammation"] = point.get("x")
                activity_lookup[key]["scatlas"] = point.get("y")

            # Convert to comparison records
            for (sig, ct), activities in activity_lookup.items():
                values = [v for v in [activities["cima"], activities["inflammation"], activities["scatlas"]] if v is not None]
                if not values:
                    continue
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5 if len(values) > 1 else 0
                cv = abs(std_val / mean_val) if mean_val != 0 else 0

                comparisons.append({
                    "signature": sig,
                    "signature_type": signature_type,
                    "cell_type": ct,
                    "cima_activity": activities["cima"],
                    "inflammation_activity": activities["inflammation"],
                    "scatlas_activity": activities["scatlas"],
                    "mean_activity": mean_val,
                    "std_activity": std_val,
                    "cv": cv,
                })
        # If it's already a list format
        elif isinstance(comparisons_raw, list):
            comparisons = self.filter_by_signature_type(comparisons_raw, signature_type)

        if cell_type:
            comparisons = [c for c in comparisons if c.get("cell_type") == cell_type]

        # Extract metadata
        cell_types = sorted(list(set(c.get("cell_type") for c in comparisons if c.get("cell_type"))))
        signatures = sorted(list(set(c.get("signature") for c in comparisons if c.get("signature"))))

        # Build summary
        correlations = await self.get_atlas_correlations(signature_type=signature_type)
        mean_corr = (
            sum(c.correlation for c in correlations) / len(correlations)
            if correlations
            else 0
        )

        conserved = await self.get_conserved_signatures(signature_type=signature_type)
        top_conserved = [c.signature for c in conserved[:5]]

        summary = CrossAtlasSummary(
            n_common_signatures=len(signatures),
            n_common_cell_types=len(cell_types),
            mean_correlation=mean_corr,
            top_conserved=top_conserved,
            top_divergent=[],  # Would need divergence analysis
            harmonization_quality=mean_corr,  # Simplified
        )

        return AtlasComparisonData(
            comparisons=[CrossAtlasComparison(**c) for c in comparisons],
            cell_types=cell_types,
            signatures=signatures,
            atlases=["CIMA", "Inflammation", "scAtlas"],
            summary=summary,
        )

    async def get_available_atlases(self) -> list[str]:
        """Get list of available atlases."""
        return ["CIMA", "Inflammation", "scAtlas"]

    # Static atlas summary metadata — the "summary" section in cross_atlas.json
    # is a simple dict (not list-of-dicts), so DuckDB flattening skips it.
    # Hard-coded here to avoid depending on the JSON file for static metadata.
    _ATLAS_SUMMARY: dict = {
        "cima": {"cells": 6484974, "samples": 421, "cell_types": 27},
        "inflammation": {"cells": 4918140, "samples": 817, "cell_types": 66},
        "scatlas_normal": {"cells": 2289588, "samples": 317, "cell_types": 376, "organs": 35, "donors": 317},
        "scatlas_cancer": {"cells": 4144933, "samples": 464, "cell_types": 156, "donors": 464},
    }

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_summary(self) -> dict:
        """
        Get atlas summary statistics for overview.

        Returns:
            Summary dict with cells, samples, cell_types per atlas
        """
        # Try loading from JSON file (nested dict format)
        try:
            data = await self.load_json("cross_atlas.json")
            if isinstance(data, dict) and "summary" in data:
                summary = data["summary"]
            else:
                # DuckDB returns flat list — summary section not captured
                summary = self._ATLAS_SUMMARY
        except Exception:
            summary = self._ATLAS_SUMMARY

        # Calculate totals
        total_cells = 0
        total_samples = 0
        total_cell_types = 0

        for atlas_key in ["cima", "inflammation", "scatlas_normal", "scatlas_cancer"]:
            atlas_data = summary.get(atlas_key, {})
            total_cells += atlas_data.get("cells", 0)
            # Use samples for CIMA/Inflammation, donors for scAtlas
            samples = atlas_data.get("samples", 0)
            donors = atlas_data.get("donors", 0)
            total_samples += samples if samples > 0 else donors
            total_cell_types += atlas_data.get("cell_types", 0)

        # Add organs info for scAtlas entries for display
        if "scatlas_normal" in summary:
            summary["scatlas_normal"]["organs"] = summary["scatlas_normal"].get("organs", 35)
        if "scatlas_cancer" in summary:
            summary["scatlas_cancer"]["organs"] = summary["scatlas_cancer"].get("organs", 0)

        return {
            "total_cells": total_cells,
            "total_samples": total_samples,
            "total_cell_types": total_cell_types,
            "atlases": summary,
            "n_atlases": 3,
            "n_signatures_cytosig": 43,
            "n_signatures_secact": 1170,
        }

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_celltype_sankey(
        self,
        level: str = "coarse",
        lineage: str | None = None,
    ) -> dict:
        """
        Get cell type mapping data for Sankey/heatmap visualization.

        Args:
            level: 'coarse' or 'fine' mapping level
            lineage: Optional lineage filter (T_cell, Myeloid, B_cell, NK_ILC)

        Returns:
            Mapping data with coarse/fine details and summary
        """
        data = await self.load_json("cross_atlas.json")
        celltype_mapping = data.get("celltype_mapping", {})

        # Get mapping data based on level
        coarse_mapping = celltype_mapping.get("coarse_mapping", [])
        fine_mapping = celltype_mapping.get("fine_mapping", [])

        # Get summary statistics
        summary = celltype_mapping.get("summary", {})

        # Apply lineage filter
        filtered_coarse = coarse_mapping
        filtered_fine = fine_mapping

        if lineage and lineage != "all":
            # Lineage mapping for coarse
            lineage_map = {
                "T_cell": ["CD4_T", "CD8_T", "Unconventional_T"],
                "Myeloid": ["Myeloid"],
                "B_cell": ["B", "Plasma"],
                "NK_ILC": ["NK"],
            }
            target_lineages = lineage_map.get(lineage, [])
            filtered_coarse = [
                d for d in coarse_mapping
                if d.get("lineage") in target_lineages
            ]

            # Lineage prefixes for fine
            lineage_prefixes = {
                "T_cell": ["CD4", "CD8", "Treg", "MAIT", "gdT", "NKT", "ILC"],
                "Myeloid": ["Mono", "Mac", "DC", "pDC", "cDC", "Mast", "Neutro", "Baso", "Eosino"],
                "B_cell": ["B_", "Plasma"],
                "NK_ILC": ["NK", "ILC"],
            }
            prefixes = lineage_prefixes.get(lineage, [])
            filtered_fine = [
                d for d in fine_mapping
                if any(
                    d.get("fine_type", "").startswith(p) or p in d.get("fine_type", "")
                    for p in prefixes
                )
            ]

        # Build Sankey nodes and links for coarse level
        nodes = []
        links = []

        if level == "coarse" and filtered_coarse:
            atlas_colors = {"CIMA": "#e41a1c", "Inflammation": "#377eb8", "scAtlas": "#4daf4a"}
            lineage_colors = {
                "CD4_T": "#984ea3", "CD8_T": "#ff7f00", "NK": "#a65628",
                "B": "#f781bf", "Plasma": "#999999", "Myeloid": "#66c2a5",
                "Unconventional_T": "#fc8d62", "Progenitor": "#8da0cb",
            }

            node_idx = {}

            # Atlas nodes
            for atlas in ["CIMA", "Inflammation", "scAtlas"]:
                node_idx[atlas] = len(nodes)
                nodes.append({"label": atlas, "color": atlas_colors[atlas], "category": "atlas"})

            # Lineage nodes
            for d in filtered_coarse:
                lin = d.get("lineage")
                node_idx[lin] = len(nodes)
                nodes.append({
                    "label": lin.replace("_", " "),
                    "color": lineage_colors.get(lin, "#999"),
                    "category": "lineage",
                })

            # Links from atlases to lineages
            atlas_keys = [
                {"key": "cima", "name": "CIMA"},
                {"key": "inflammation", "name": "Inflammation"},
                {"key": "scatlas", "name": "scAtlas"},
            ]
            for d in filtered_coarse:
                lin = d.get("lineage")
                for ak in atlas_keys:
                    atlas_data = d.get(ak["key"], {})
                    total_cells = atlas_data.get("total_cells", 0)
                    if total_cells > 0:
                        import math
                        scaled_value = math.log10(total_cells + 1) * 10
                        types_list = atlas_data.get("types", [])
                        links.append({
                            "source": node_idx[ak["name"]],
                            "target": node_idx[lin],
                            "value": max(scaled_value, 1),
                            "cells": total_cells,
                            "types": ", ".join(t.get("name", "") for t in types_list[:5]),
                            "n_types": len(types_list),
                        })

        # Sort fine mapping by total cells
        if filtered_fine:
            for d in filtered_fine:
                d["total_cells"] = (
                    d.get("cima", {}).get("total_cells", 0) +
                    d.get("inflammation", {}).get("total_cells", 0) +
                    d.get("scatlas", {}).get("total_cells", 0)
                )
            filtered_fine = sorted(filtered_fine, key=lambda x: x.get("total_cells", 0), reverse=True)

        return {
            "nodes": nodes,
            "links": links,
            "coarse_mapping": filtered_coarse,
            "fine_mapping": filtered_fine,
            "summary": summary,
            "level": level,
            "lineage_filter": lineage,
            "available_lineages": [
                {"value": "all", "label": "All Lineages"},
                {"value": "T_cell", "label": "T Cells (CD4, CD8, Unconventional)"},
                {"value": "Myeloid", "label": "Myeloid (Mono, DC, Mac)"},
                {"value": "B_cell", "label": "B Cells & Plasma"},
                {"value": "NK_ILC", "label": "NK & ILC"},
            ],
        }

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_pairwise_scatter(
        self,
        atlas1: str = "CIMA",
        atlas2: str = "Inflammation",
        signature_type: str = "CytoSig",
        level: str = "coarse",
        view: str = "pseudobulk",
    ) -> dict:
        """
        Get pairwise scatter plot data for atlas comparison.

        Args:
            atlas1: First atlas name
            atlas2: Second atlas name
            signature_type: 'CytoSig' or 'SecAct'
            level: 'coarse' or 'fine' aggregation level
            view: 'pseudobulk' for sample-aggregated, 'singlecell' for cell-level

        Returns:
            Scatter data with correlation statistics
        """
        data = await self.load_json("cross_atlas.json")
        atlas_comparison = data.get("atlas_comparison", {})

        # Get the right signature type section
        sig_key = signature_type.lower()
        sig_data = atlas_comparison.get(sig_key, {})

        # Map atlas names to pair keys
        pair_map = {
            ("CIMA", "Inflammation"): "cima_vs_inflammation",
            ("Inflammation", "CIMA"): "cima_vs_inflammation",
            ("CIMA", "scAtlas"): "cima_vs_scatlas",
            ("scAtlas", "CIMA"): "cima_vs_scatlas",
            ("Inflammation", "scAtlas"): "inflammation_vs_scatlas",
            ("scAtlas", "Inflammation"): "inflammation_vs_scatlas",
        }

        pair_key = pair_map.get((atlas1, atlas2))
        if not pair_key:
            return {"error": f"Invalid atlas pair: {atlas1} vs {atlas2}"}

        pair_data = sig_data.get(pair_key, {})

        # Get the right data based on view type
        if view == "singlecell":
            level_key = f"singlecell_mean_{level}"
        else:
            level_key = f"celltype_aggregated_{level}"

        level_data = pair_data.get(level_key, {})

        # Determine if we need to swap x/y based on atlas order
        swap_axes = (atlas1, atlas2) in [
            ("Inflammation", "CIMA"),
            ("scAtlas", "CIMA"),
            ("scAtlas", "Inflammation"),
        ]

        scatter_data = level_data.get("data", [])
        if swap_axes:
            # Swap x/y and also swap n_samples/n_cells fields
            swapped_data = []
            for point in scatter_data:
                swapped_point = {
                    **point,
                    "x": point.get("y"),
                    "y": point.get("x"),
                }
                # Handle both pseudobulk (n_samples) and single-cell (n_cells) fields
                if "n_samples_x" in point:
                    swapped_point["n_samples_x"] = point.get("n_samples_y")
                    swapped_point["n_samples_y"] = point.get("n_samples_x")
                if "n_cells_x" in point:
                    swapped_point["n_cells_x"] = point.get("n_cells_y")
                    swapped_point["n_cells_y"] = point.get("n_cells_x")
                swapped_data.append(swapped_point)
            scatter_data = swapped_data

        # Build response - handle different field names for correlation
        correlation = level_data.get("correlation") or level_data.get("overall_correlation", 0)
        pvalue = level_data.get("pvalue") or level_data.get("overall_pvalue", 1)

        result = {
            "data": scatter_data,
            "correlation": correlation,
            "pvalue": pvalue,
            "n": level_data.get("n", 0),
            "n_celltypes": level_data.get("n_celltypes", 0),
            "n_signatures": level_data.get("n_signatures", 0),
            "atlas1": atlas1,
            "atlas2": atlas2,
            "signature_type": signature_type,
            "level": level,
            "view": view,
        }

        # Add single-cell specific fields
        if view == "singlecell":
            n_cells_1 = level_data.get("n_cells_atlas1", 0)
            n_cells_2 = level_data.get("n_cells_atlas2", 0)
            if swap_axes:
                n_cells_1, n_cells_2 = n_cells_2, n_cells_1
            result["n_cells_atlas1"] = n_cells_1
            result["n_cells_atlas2"] = n_cells_2

        return result

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_signature_reliability(
        self,
        signature_type: str = "CytoSig",
    ) -> dict:
        """
        Get signature reliability data for heatmap.

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            Signature reliability with per-pair correlations
        """
        data = await self.load_json("cross_atlas.json")
        sig_reliability = data.get("signature_reliability", {})

        sig_key = signature_type.lower()
        reliability_data = sig_reliability.get(sig_key, {})

        signatures = reliability_data.get("signatures", [])
        summary = reliability_data.get("summary", {})
        pair_correlations = reliability_data.get("pair_correlations", {})

        # Build heatmap matrix data
        # Rows: signatures, Columns: atlas pairs
        pairs = ["cima_vs_inflammation", "cima_vs_scatlas", "inflammation_vs_scatlas"]
        pair_labels = ["CIMA vs Inflammation", "CIMA vs scAtlas", "Inflammation vs scAtlas"]

        heatmap_data = []
        for sig in signatures:
            row = {
                "signature": sig.get("signature"),
                "category": sig.get("category"),
                "mean_correlation": sig.get("mean_correlation"),
                "correlations": {},
            }
            correlations = sig.get("correlations", {})
            for pair in pairs:
                pair_corr = correlations.get(pair, {})
                row["correlations"][pair] = {
                    "r": pair_corr.get("r", None),
                    "p": pair_corr.get("p", None),
                    "n": pair_corr.get("n", None),
                }
            heatmap_data.append(row)

        return {
            "signatures": heatmap_data,
            "summary": summary,
            "pair_labels": pair_labels,
            "pairs": pairs,
            "signature_type": signature_type,
        }

    @cached(prefix="cross_atlas", ttl=3600)
    async def get_meta_analysis_forest(
        self,
        analysis: str = "age",
        signature_type: str = "CytoSig",
        signature: str | None = None,
    ) -> dict:
        """
        Get meta-analysis data for forest plot.

        Args:
            analysis: 'age', 'bmi', or 'sex'
            signature_type: 'CytoSig' or 'SecAct'
            signature: Optional specific signature filter

        Returns:
            Forest plot data with individual and pooled effects
        """
        data = await self.load_json("cross_atlas.json")
        meta_analysis = data.get("meta_analysis", {})

        analysis_data = meta_analysis.get(analysis, [])

        # Filter by signature type
        sig_key = "CytoSig" if signature_type == "CytoSig" else "SecAct"
        filtered = [
            r for r in analysis_data
            if r.get("sig_type") == sig_key
        ]

        # Filter by specific signature if provided
        if signature:
            filtered = [r for r in filtered if r.get("signature") == signature]

        # Group by signature for forest plot
        signature_groups = {}
        for r in filtered:
            sig = r.get("signature")
            if sig not in signature_groups:
                signature_groups[sig] = {
                    "signature": sig,
                    "individual_effects": [],
                    "pooled_effect": r.get("pooled_effect"),
                    "pooled_se": r.get("pooled_se"),
                    "ci_low": r.get("ci_low"),
                    "ci_high": r.get("ci_high"),
                    "I2": r.get("I2"),
                    "n_atlases": r.get("n_atlases"),
                }
            signature_groups[sig]["individual_effects"].append({
                "atlas": r.get("atlas"),
                "effect": r.get("effect"),
                "se": r.get("se"),
                "pvalue": r.get("pvalue"),
                "n": r.get("n"),
            })

        # Convert to list and sort by pooled effect
        forest_data = list(signature_groups.values())
        forest_data.sort(key=lambda x: abs(x.get("pooled_effect", 0)), reverse=True)

        # Calculate summary statistics
        n_significant = sum(
            1 for s in forest_data
            if s.get("pooled_effect") and abs(s.get("pooled_effect", 0)) > 0.1
        )
        n_consistent = sum(
            1 for s in forest_data
            if all(
                e.get("effect", 0) * s.get("pooled_effect", 0) > 0
                for e in s.get("individual_effects", [])
                if e.get("effect") is not None
            )
        )
        n_heterogeneous = sum(
            1 for s in forest_data
            if s.get("I2", 0) > 50
        )

        return {
            "analysis": analysis,
            "signature_type": signature_type,
            "forest_data": forest_data,
            "summary": {
                "n_signatures": len(forest_data),
                "n_significant": n_significant,
                "n_consistent_direction": n_consistent,
                "n_heterogeneous": n_heterogeneous,
            },
        }
