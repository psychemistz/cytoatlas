"""
Gene name mapping between HGNC and CytoSig conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json


# CytoSig to HGNC mapping (CytoSig names differ from standard gene symbols)
CYTOSIG_TO_HGNC = {
    "TNFA": "TNF",
    "TGFB1": "TGFB1",
    "TGFB2": "TGFB2",
    "TGFB3": "TGFB3",
    "IFNG": "IFNG",
    "IFNA": "IFNA1",  # IFNA represents multiple genes
    "IFNB": "IFNB1",
    "IL1A": "IL1A",
    "IL1B": "IL1B",
    "IL2": "IL2",
    "IL4": "IL4",
    "IL6": "IL6",
    "IL10": "IL10",
    "IL12": "IL12A",  # IL12 is p35 + p40
    "IL13": "IL13",
    "IL15": "IL15",
    "IL17A": "IL17A",
    "IL17F": "IL17F",
    "IL18": "IL18",
    "IL21": "IL21",
    "IL22": "IL22",
    "IL23": "IL23A",  # IL23 is p19 + p40
    "IL27": "IL27",
    "IL33": "IL33",
    "CSF1": "CSF1",
    "CSF2": "CSF2",
    "CSF3": "CSF3",
    "CCL2": "CCL2",
    "CCL3": "CCL3",
    "CCL4": "CCL4",
    "CCL5": "CCL5",
    "CCL7": "CCL7",
    "CCL8": "CCL8",
    "CXCL1": "CXCL1",
    "CXCL2": "CXCL2",
    "CXCL8": "CXCL8",
    "CXCL9": "CXCL9",
    "CXCL10": "CXCL10",
    "CXCL11": "CXCL11",
    "CXCL12": "CXCL12",
    "VEGFA": "VEGFA",
    "EGF": "EGF",
    "FGF2": "FGF2",
    "HGF": "HGF",
}

# Reverse mapping
HGNC_TO_CYTOSIG = {v: k for k, v in CYTOSIG_TO_HGNC.items()}


class GeneMapper:
    """Maps between gene naming conventions."""

    def __init__(
        self,
        cytosig_to_hgnc: Optional[dict[str, str]] = None,
        custom_mapping: Optional[dict[str, str]] = None,
    ):
        self.cytosig_to_hgnc = cytosig_to_hgnc or CYTOSIG_TO_HGNC.copy()
        self.hgnc_to_cytosig = {v: k for k, v in self.cytosig_to_hgnc.items()}

        if custom_mapping:
            self.cytosig_to_hgnc.update(custom_mapping)
            self.hgnc_to_cytosig.update({v: k for k, v in custom_mapping.items()})

    def to_hgnc(self, name: str) -> str:
        """Convert CytoSig name to HGNC symbol."""
        return self.cytosig_to_hgnc.get(name, name)

    def to_cytosig(self, name: str) -> str:
        """Convert HGNC symbol to CytoSig name."""
        return self.hgnc_to_cytosig.get(name, name)

    def normalize(self, name: str) -> str:
        """Normalize gene name to HGNC."""
        return self.to_hgnc(name)

    def get_aliases(self, name: str) -> list[str]:
        """Get all known aliases for a gene name."""
        aliases = [name]

        # Add HGNC version
        hgnc = self.to_hgnc(name)
        if hgnc != name:
            aliases.append(hgnc)

        # Add CytoSig version
        cytosig = self.to_cytosig(name)
        if cytosig != name:
            aliases.append(cytosig)

        return list(set(aliases))

    def map_list(
        self,
        names: list[str],
        direction: str = "to_hgnc",
    ) -> list[str]:
        """Map a list of gene names.

        Parameters
        ----------
        names : list[str]
            Gene names to map
        direction : str
            "to_hgnc" or "to_cytosig"

        Returns
        -------
        list[str]
            Mapped gene names
        """
        if direction == "to_hgnc":
            return [self.to_hgnc(n) for n in names]
        else:
            return [self.to_cytosig(n) for n in names]

    def load_mapping(self, path: Path) -> None:
        """Load additional mapping from JSON file."""
        with open(path) as f:
            custom = json.load(f)

        self.cytosig_to_hgnc.update(custom)
        self.hgnc_to_cytosig.update({v: k for k, v in custom.items()})

    def save_mapping(self, path: Path) -> None:
        """Save mapping to JSON file."""
        with open(path, "w") as f:
            json.dump(self.cytosig_to_hgnc, f, indent=2)


def get_gene_mapping() -> dict[str, str]:
    """Get default CytoSig to HGNC mapping."""
    return CYTOSIG_TO_HGNC.copy()
