"""Perturbation data service for parse_10M and Tahoe datasets."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import cached
from app.services.base import BaseService

settings = get_settings()


class PerturbationService(BaseService):
    """Service for perturbation data (parse_10M cytokine stimulation + Tahoe drug response)."""

    def __init__(self, db: AsyncSession | None = None):
        super().__init__(db)
        self.data_dir = settings.viz_data_path

    # -----------------------------------------------------------------------
    #  parse_10M methods — cytokine stimulation perturbation screen
    # -----------------------------------------------------------------------

    @cached(prefix="perturbation", ttl=3600)
    async def get_parse10m_summary(self) -> dict:
        """
        Get parse_10M dataset summary statistics.

        Returns:
            Dict with n_cells, n_donors, n_cytokines, n_cell_types,
            signature_types, and dataset metadata.
        """
        data = await self.load_json("parse10m_cytokine_heatmap.json")

        # Extract metadata from heatmap structure
        if isinstance(data, dict):
            return {
                "dataset": "parse_10M",
                "description": "Cytokine stimulation perturbation screen",
                "n_cytokines": len(data.get("cytokines", [])),
                "n_cell_types": len(data.get("cell_types", [])),
                "cytokines": data.get("cytokines", []),
                "cell_types": data.get("cell_types", []),
                "signature_types": data.get("signature_types", ["CytoSig", "SecAct"]),
            }

        # Flat list format — derive metadata from records
        cytokines = sorted(set(r.get("cytokine") for r in data if r.get("cytokine")))
        cell_types = sorted(set(r.get("cell_type") for r in data if r.get("cell_type")))

        return {
            "dataset": "parse_10M",
            "description": "Cytokine stimulation perturbation screen",
            "n_cytokines": len(cytokines),
            "n_cell_types": len(cell_types),
            "cytokines": cytokines,
            "cell_types": cell_types,
            "signature_types": ["CytoSig", "SecAct"],
        }

    @cached(prefix="perturbation", ttl=3600)
    async def get_cytokine_response(
        self,
        cytokine: str | None = None,
        cell_type: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get cytokine response activity from the heatmap data.

        Args:
            cytokine: Optional cytokine filter (e.g., 'IL6', 'IFNG')
            cell_type: Optional cell type filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of cytokine response records with activity values.
        """
        data = await self.load_json("parse10m_cytokine_heatmap.json")

        # Handle nested dict vs flat list
        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if cytokine:
            results = [r for r in results if r.get("cytokine") == cytokine]

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_ground_truth_validation(
        self,
        signature_type: str = "CytoSig",
        cytokine: str | None = None,
        cell_type: str | None = None,
    ) -> list[dict]:
        """
        Get ground truth validation of activity predictions against known perturbations.

        Compares inferred activity signatures with actual cytokine stimulation
        to validate prediction accuracy.

        Args:
            signature_type: 'CytoSig' or 'SecAct'
            cytokine: Optional cytokine filter
            cell_type: Optional cell type filter

        Returns:
            List of ground truth validation records with predicted vs actual.
        """
        data = await self.load_json("parse10m_ground_truth.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if cytokine:
            results = [r for r in results if r.get("cytokine") == cytokine]

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_treatment_effect_heatmap(
        self,
        cell_type: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get treatment effect heatmap showing activity changes across cytokines.

        Args:
            cell_type: Optional cell type filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of heatmap records (cytokine x signature activity matrix).
        """
        data = await self.load_json("parse10m_cytokine_heatmap.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_cytokine_families(self) -> list[dict]:
        """
        Get cytokine family groupings used in the parse_10M screen.

        Returns:
            List of dicts with cytokine, family, and subfamily fields.
        """
        data = await self.load_json("parse10m_cytokine_heatmap.json")

        if isinstance(data, dict):
            families = data.get("cytokine_families", [])
            if families:
                return families

            # Derive from cytokine list if families not stored separately
            cytokines = data.get("cytokines", [])
            return [{"cytokine": c, "family": None, "subfamily": None} for c in cytokines]

        # Flat list — extract unique cytokines
        cytokines = sorted(set(r.get("cytokine") for r in data if r.get("cytokine")))
        return [{"cytokine": c, "family": None, "subfamily": None} for c in cytokines]

    @cached(prefix="perturbation", ttl=3600)
    async def get_donor_variability(
        self,
        cytokine: str | None = None,
        cell_type: str | None = None,
    ) -> list[dict]:
        """
        Get donor-level variability in cytokine response.

        Args:
            cytokine: Optional cytokine filter
            cell_type: Optional cell type filter

        Returns:
            List of donor variability records with variance, CV, and per-donor activity.
        """
        data = await self.load_json("parse10m_donor_variability.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        if cytokine:
            results = [r for r in results if r.get("cytokine") == cytokine]

        if cell_type:
            results = self.filter_by_cell_type(results, cell_type)

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_parse10m_cytokines(self) -> list[str]:
        """
        Get list of available cytokines in the parse_10M dataset.

        Returns:
            Sorted list of cytokine names.
        """
        data = await self.load_json("parse10m_cytokine_heatmap.json")

        if isinstance(data, dict):
            cytokines = data.get("cytokines", [])
            if cytokines:
                return sorted(cytokines)
            records = data.get("data", [])
        else:
            records = data

        return sorted(set(r.get("cytokine") for r in records if r.get("cytokine")))

    @cached(prefix="perturbation", ttl=3600)
    async def get_parse10m_cell_types(self) -> list[str]:
        """
        Get list of available cell types in the parse_10M dataset.

        Returns:
            Sorted list of cell type names.
        """
        data = await self.load_json("parse10m_cytokine_heatmap.json")

        if isinstance(data, dict):
            cell_types = data.get("cell_types", [])
            if cell_types:
                return sorted(cell_types)
            records = data.get("data", [])
        else:
            records = data

        return sorted(set(r.get("cell_type") for r in records if r.get("cell_type")))

    # -----------------------------------------------------------------------
    #  Tahoe methods — drug response perturbation screen
    # -----------------------------------------------------------------------

    @cached(prefix="perturbation", ttl=3600)
    async def get_tahoe_summary(self) -> dict:
        """
        Get Tahoe dataset summary statistics.

        Returns:
            Dict with n_drugs, n_cell_lines, drug list, cell line list,
            and dataset metadata.
        """
        data = await self.load_json("tahoe_drug_sensitivity.json")

        if isinstance(data, dict):
            return {
                "dataset": "Tahoe",
                "description": "Drug response perturbation screen",
                "n_drugs": len(data.get("drugs", [])),
                "n_cell_lines": len(data.get("cell_lines", [])),
                "drugs": data.get("drugs", []),
                "cell_lines": data.get("cell_lines", []),
                "signature_types": data.get("signature_types", ["CytoSig", "SecAct"]),
            }

        drugs = sorted(set(r.get("drug") for r in data if r.get("drug")))
        cell_lines = sorted(set(r.get("cell_line") for r in data if r.get("cell_line")))

        return {
            "dataset": "Tahoe",
            "description": "Drug response perturbation screen",
            "n_drugs": len(drugs),
            "n_cell_lines": len(cell_lines),
            "drugs": drugs,
            "cell_lines": cell_lines,
            "signature_types": ["CytoSig", "SecAct"],
        }

    @cached(prefix="perturbation", ttl=3600)
    async def get_drug_response(
        self,
        drug: str | None = None,
        cell_line: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get drug response activity signatures.

        Args:
            drug: Optional drug name filter
            cell_line: Optional cell line filter
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of drug response records with activity values.
        """
        data = await self.load_json("tahoe_drug_sensitivity.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        if drug:
            results = [r for r in results if r.get("drug") == drug]

        if cell_line:
            results = [r for r in results if r.get("cell_line") == cell_line]

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_drug_sensitivity_matrix(
        self,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        """
        Get drug sensitivity matrix (drugs x signatures).

        Args:
            signature_type: 'CytoSig' or 'SecAct'

        Returns:
            List of sensitivity matrix records.
        """
        data = await self.load_json("tahoe_drug_sensitivity.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        results = self.filter_by_signature_type(results, signature_type)

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_dose_response(
        self,
        drug: str | None = None,
        cell_line: str | None = None,
    ) -> list[dict]:
        """
        Get dose-response curves for drug treatments.

        Args:
            drug: Optional drug name filter
            cell_line: Optional cell line filter

        Returns:
            List of dose-response records with dose, activity, and viability.
        """
        data = await self.load_json("tahoe_dose_response.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        if drug:
            results = [r for r in results if r.get("drug") == drug]

        if cell_line:
            results = [r for r in results if r.get("cell_line") == cell_line]

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_pathway_activation(
        self,
        drug: str | None = None,
    ) -> list[dict]:
        """
        Get pathway activation signatures in response to drug treatment.

        Args:
            drug: Optional drug name filter

        Returns:
            List of pathway activation records with pathway, activity, and p-value.
        """
        data = await self.load_json("tahoe_pathway_activation.json")

        if isinstance(data, dict):
            results = data.get("data", [])
        else:
            results = data

        if drug:
            results = [r for r in results if r.get("drug") == drug]

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_cell_line_profiles(
        self,
        cell_line: str | None = None,
    ) -> list[dict]:
        """
        Get baseline activity profiles for cell lines.

        Args:
            cell_line: Optional cell line filter

        Returns:
            List of cell line profile records with baseline activity signatures.
        """
        data = await self.load_json("tahoe_drug_sensitivity.json")

        if isinstance(data, dict):
            results = data.get("cell_line_profiles", [])
            if not results:
                # Fall back to extracting unique cell line records
                results = data.get("data", [])
        else:
            results = data

        if cell_line:
            results = [r for r in results if r.get("cell_line") == cell_line]

        return results

    @cached(prefix="perturbation", ttl=3600)
    async def get_tahoe_drugs(self) -> list[str]:
        """
        Get list of available drugs in the Tahoe dataset.

        Returns:
            Sorted list of drug names.
        """
        data = await self.load_json("tahoe_drug_sensitivity.json")

        if isinstance(data, dict):
            drugs = data.get("drugs", [])
            if drugs:
                return sorted(drugs)
            records = data.get("data", [])
        else:
            records = data

        return sorted(set(r.get("drug") for r in records if r.get("drug")))

    @cached(prefix="perturbation", ttl=3600)
    async def get_tahoe_cell_lines(self) -> list[str]:
        """
        Get list of available cell lines in the Tahoe dataset.

        Returns:
            Sorted list of cell line names.
        """
        data = await self.load_json("tahoe_drug_sensitivity.json")

        if isinstance(data, dict):
            cell_lines = data.get("cell_lines", [])
            if cell_lines:
                return sorted(cell_lines)
            records = data.get("data", [])
        else:
            records = data

        return sorted(set(r.get("cell_line") for r in records if r.get("cell_line")))

    # -----------------------------------------------------------------------
    #  Combined perturbation summary
    # -----------------------------------------------------------------------

    @cached(prefix="perturbation", ttl=3600)
    async def get_perturbation_summary(self) -> dict:
        """
        Get combined summary across all perturbation datasets.

        Returns:
            Dict with parse_10M and Tahoe summaries plus combined statistics.
        """
        parse10m_summary = await self.get_parse10m_summary()
        tahoe_summary = await self.get_tahoe_summary()

        return {
            "parse_10M": parse10m_summary,
            "tahoe": tahoe_summary,
            "total_datasets": 2,
            "signature_types": ["CytoSig", "SecAct"],
        }

    # -----------------------------------------------------------------------
    #  Router-facing aliases
    #  (Routers call these names; delegate to the canonical methods above)
    # -----------------------------------------------------------------------

    async def get_summary(self) -> dict:
        return await self.get_perturbation_summary()

    async def get_parse10m_activity(
        self,
        cytokine: str | None = None,
        cell_type: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_cytokine_response(
            cytokine=cytokine, cell_type=cell_type, signature_type=signature_type,
        )

    async def get_parse10m_treatment_effect(
        self,
        cell_type: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_treatment_effect_heatmap(
            cell_type=cell_type, signature_type=signature_type,
        )

    async def get_parse10m_ground_truth(
        self,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_ground_truth_validation(signature_type=signature_type)

    async def get_parse10m_heatmap(
        self,
        cell_type: str | None = None,
        signature_type: str = "CytoSig",
    ) -> dict:
        data = await self.get_treatment_effect_heatmap(
            cell_type=cell_type, signature_type=signature_type,
        )
        return data

    async def get_parse10m_donor_variability(
        self,
        cytokine: str | None = None,
        cell_type: str | None = None,
    ) -> list[dict]:
        return await self.get_donor_variability(
            cytokine=cytokine, cell_type=cell_type,
        )

    async def get_parse10m_cytokine_families(self) -> list[dict]:
        return await self.get_cytokine_families()

    async def get_tahoe_activity(
        self,
        drug: str | None = None,
        cell_line: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_drug_response(
            drug=drug, cell_line=cell_line, signature_type=signature_type,
        )

    async def get_tahoe_drug_effect(
        self,
        cell_line: str | None = None,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_drug_response(
            cell_line=cell_line, signature_type=signature_type,
        )

    async def get_tahoe_sensitivity_matrix(
        self,
        signature_type: str = "CytoSig",
    ) -> list[dict]:
        return await self.get_drug_sensitivity_matrix(signature_type=signature_type)

    async def get_tahoe_dose_response(
        self,
        drug: str | None = None,
        cell_line: str | None = None,
    ) -> list[dict]:
        return await self.get_dose_response(drug=drug, cell_line=cell_line)

    async def get_tahoe_pathway_activation(
        self,
        drug: str | None = None,
    ) -> list[dict]:
        return await self.get_pathway_activation(drug=drug)
