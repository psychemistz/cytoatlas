"""
Signature matrix loaders.

Provides access to CytoSig, SecAct, and custom signature matrices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# Add SecActpy to path
SECACTPY_PATH = Path("/vf/users/parks34/projects/1ridgesig/SecActpy")
if str(SECACTPY_PATH) not in sys.path:
    sys.path.insert(0, str(SECACTPY_PATH))


class SignatureLoader:
    """
    Loads and manages signature matrices.

    Example:
        >>> loader = SignatureLoader()
        >>> cytosig = loader.get("CytoSig")
        >>> secact = loader.get("SecAct")
        >>> lincytosig = loader.get("LinCytoSig")
        >>> custom = loader.load_custom("/path/to/signature.csv")
    """

    # Built-in signature paths (relative to SecActpy)
    BUILTIN_SIGNATURES = {
        "CytoSig": "data/CytoSig.signature.csv",
        "SecAct": "data/SecAct.signature.csv",
    }

    # External signatures with absolute paths
    EXTERNAL_SIGNATURES = {
        "LinCytoSig": Path("/data/parks34/projects/2secactpy/results/celltype_signatures/celltype_cytokine_signatures.csv"),
    }

    def __init__(self, secactpy_path: Optional[Path] = None):
        """
        Initialize signature loader.

        Args:
            secactpy_path: Path to SecActpy package.
        """
        self.secactpy_path = secactpy_path or SECACTPY_PATH
        self._cache: dict[str, pd.DataFrame] = {}

    def get(self, name: str) -> pd.DataFrame:
        """
        Get a signature matrix by name.

        Args:
            name: Signature name ("CytoSig", "SecAct", or "LinCytoSig").

        Returns:
            Signature matrix (genes x signatures).
        """
        if name in self._cache:
            return self._cache[name]

        if name == "CytoSig":
            sig = load_cytosig()
        elif name == "SecAct":
            sig = load_secact()
        elif name == "LinCytoSig":
            sig = load_lincytosig()
        elif name in self.BUILTIN_SIGNATURES:
            path = self.secactpy_path / self.BUILTIN_SIGNATURES[name]
            sig = pd.read_csv(path, index_col=0)
        elif name in self.EXTERNAL_SIGNATURES:
            path = self.EXTERNAL_SIGNATURES[name]
            sig = pd.read_csv(path, index_col=0)
        else:
            raise ValueError(f"Unknown signature: {name}")

        self._cache[name] = sig
        return sig

    def load_custom(
        self,
        path: Union[str, Path],
        gene_col: Optional[str] = None,
        sep: str = ",",
    ) -> pd.DataFrame:
        """
        Load a custom signature matrix.

        Args:
            path: Path to signature file.
            gene_col: Column containing gene names (uses index if None).
            sep: Column separator.

        Returns:
            Signature matrix (genes x signatures).
        """
        path = Path(path)

        if path.suffix == ".csv":
            sig = pd.read_csv(path, sep=sep, index_col=0 if gene_col is None else None)
        elif path.suffix == ".tsv":
            sig = pd.read_csv(path, sep="\t", index_col=0 if gene_col is None else None)
        elif path.suffix == ".parquet":
            sig = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        if gene_col is not None:
            sig = sig.set_index(gene_col)

        return sig

    def list_available(self) -> list[str]:
        """List available signatures (built-in and external)."""
        return list(self.BUILTIN_SIGNATURES.keys()) + list(self.EXTERNAL_SIGNATURES.keys())

    def get_signature_info(self, name: str) -> dict:
        """
        Get information about a signature.

        Args:
            name: Signature name.

        Returns:
            Dict with signature metadata.
        """
        sig = self.get(name)
        return {
            "name": name,
            "n_genes": len(sig),
            "n_signatures": sig.shape[1],
            "signatures": list(sig.columns),
            "genes_sample": list(sig.index[:10]),
        }


def load_cytosig() -> pd.DataFrame:
    """
    Load CytoSig signature matrix.

    Returns:
        CytoSig matrix (genes x 44 cytokines).
    """
    try:
        from secactpy import load_cytosig as _load_cytosig
        return _load_cytosig()
    except ImportError:
        # Fallback to direct file load
        path = SECACTPY_PATH / "data" / "CytoSig.signature.csv"
        if path.exists():
            return pd.read_csv(path, index_col=0)
        raise ImportError(
            "Could not load CytoSig. Ensure SecActpy is installed."
        )


def load_secact() -> pd.DataFrame:
    """
    Load SecAct signature matrix.

    Returns:
        SecAct matrix (genes x 1,249 secreted proteins).
    """
    try:
        from secactpy import load_secact as _load_secact
        return _load_secact()
    except ImportError:
        # Fallback to direct file load
        path = SECACTPY_PATH / "data" / "SecAct.signature.csv"
        if path.exists():
            return pd.read_csv(path, index_col=0)
        raise ImportError(
            "Could not load SecAct. Ensure SecActpy is installed."
        )


# Default path for LinCytoSig
LINCYTOSIG_PATH = Path("/data/parks34/projects/2secactpy/results/celltype_signatures/celltype_cytokine_signatures.csv")


def load_lincytosig(
    path: Optional[Union[str, Path]] = None,
    cell_types: Optional[list[str]] = None,
    cytokines: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load LinCytoSig signature matrix.

    LinCytoSig contains cell-type-specific cytokine response signatures,
    derived from the CytoSig differential expression database using
    median aggregation across experiments.

    Signature columns are named as "CellType__Cytokine" (e.g., "Macrophage__IFNG").

    Args:
        path: Path to LinCytoSig CSV file. If None, uses default location.
        cell_types: Filter to specific cell types (e.g., ["Macrophage", "T_CD4"]).
        cytokines: Filter to specific cytokines (e.g., ["IFNG", "TNFA"]).

    Returns:
        LinCytoSig matrix (genes x cell_type__cytokine signatures).
    """
    try:
        from secactpy import load_lincytosig as _load_lincytosig
        return _load_lincytosig(path=path, cell_types=cell_types, cytokines=cytokines)
    except ImportError:
        # Fallback to direct file load
        if path is None:
            path = LINCYTOSIG_PATH
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"LinCytoSig file not found: {path}\n"
                "Generate it using: python scripts/build_celltype_signatures.py"
            )

        df = pd.read_csv(path, index_col=0)

        # Filter by cell types
        if cell_types is not None:
            matching_cols = [
                col for col in df.columns
                if col.split('__')[0] in cell_types
            ]
            df = df[matching_cols]

        # Filter by cytokines
        if cytokines is not None:
            matching_cols = [
                col for col in df.columns
                if col.split('__')[1] in cytokines
            ]
            df = df[matching_cols]

        return df


def load_custom_signature(
    path: Union[str, Path],
    gene_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a custom signature matrix.

    Args:
        path: Path to signature file (CSV, TSV, or Parquet).
        gene_col: Column containing gene names (index if None).

    Returns:
        Signature matrix (genes x signatures).
    """
    loader = SignatureLoader()
    return loader.load_custom(path, gene_col=gene_col)


def get_signature_genes(name: str = "CytoSig") -> list[str]:
    """
    Get list of genes in a signature.

    Args:
        name: Signature name.

    Returns:
        List of gene names.
    """
    loader = SignatureLoader()
    sig = loader.get(name)
    return list(sig.index)


def get_signature_names(name: str = "CytoSig") -> list[str]:
    """
    Get list of signatures (columns).

    Args:
        name: Signature matrix name.

    Returns:
        List of signature/protein names.
    """
    loader = SignatureLoader()
    sig = loader.get(name)
    return list(sig.columns)
