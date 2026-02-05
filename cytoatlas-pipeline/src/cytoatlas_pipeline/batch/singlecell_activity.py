"""
GPU-optimized single-cell activity inference.

Streams H5AD files and processes cells in batches using CuPy GPU acceleration.
Supports CytoSig and SecAct signatures with memory-efficient processing.

Key Features:
- H5AD streaming (backed mode) for memory efficiency
- Configurable batch sizes (default: 10,000 cells)
- CuPy GPU acceleration with NumPy fallback
- Incremental result writing
- Support for CytoSig and SecAct signatures
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

# Try importing CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Default parameters
DEFAULT_LAMBDA = 5e5
DEFAULT_NRAND = 1000
DEFAULT_SEED = 42
EPS = 1e-10


def log(msg: str):
    """Print timestamped log message."""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


@dataclass
class SignatureInfo:
    """Signature matrix information."""
    name: str
    matrix: np.ndarray  # (n_genes, n_features) - genes x signatures
    gene_names: List[str]
    feature_names: List[str]
    n_genes: int
    n_features: int


def load_signature_matrix(
    signature_type: Literal["cytosig", "secact", "lincytosig"],
    genes: Optional[List[str]] = None,
) -> SignatureInfo:
    """
    Load signature matrix for activity inference.

    Args:
        signature_type: "cytosig", "secact", or "lincytosig"
        genes: Optional list of genes to subset (for matching with expression data)

    Returns:
        SignatureInfo with matrix and metadata
    """
    from secactpy import load_cytosig, load_secact

    if signature_type.lower() == "cytosig":
        sig_df = load_cytosig()
    elif signature_type.lower() == "secact":
        sig_df = load_secact()
    elif signature_type.lower() == "lincytosig":
        # Load LinCytoSig from file
        import gzip
        lincytosig_path = Path("/vf/users/parks34/projects/1ridgesig/SecActpy-dev/secactpy/data/LinCytoSig.tsv.gz")
        with gzip.open(lincytosig_path, 'rt') as f:
            sig_df = pd.read_csv(f, sep='\t', index_col=0)
    else:
        raise ValueError(f"Unknown signature type: {signature_type}")

    # Subset to requested genes if provided
    if genes is not None:
        # Find intersection
        available_genes = set(sig_df.index) & set(genes)
        sig_df = sig_df.loc[list(available_genes)]

    return SignatureInfo(
        name=signature_type,
        matrix=sig_df.values.astype(np.float64),
        gene_names=list(sig_df.index),
        feature_names=list(sig_df.columns),
        n_genes=sig_df.shape[0],
        n_features=sig_df.shape[1],
    )


def _compute_projection_matrix(
    X: np.ndarray,
    lambda_: float,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute projection matrix T = (X'X + λI)^{-1} X'.

    Args:
        X: Signature matrix (n_genes, n_features)
        lambda_: Ridge regularization parameter
        use_gpu: Use GPU if available

    Returns:
        T: Projection matrix (n_features, n_genes)
    """
    n_genes, n_features = X.shape

    if use_gpu and CUPY_AVAILABLE:
        X_gpu = cp.asarray(X, dtype=cp.float64)
        XtX = X_gpu.T @ X_gpu
        XtX_reg = XtX + lambda_ * cp.eye(n_features, dtype=cp.float64)

        try:
            XtX_inv = cp.linalg.inv(XtX_reg)
        except cp.linalg.LinAlgError:
            XtX_inv = cp.linalg.pinv(XtX_reg)

        T = XtX_inv @ X_gpu.T
        T = cp.asnumpy(T)

        del X_gpu, XtX, XtX_reg, XtX_inv
        cp.get_default_memory_pool().free_all_blocks()
    else:
        from scipy import linalg
        XtX = X.T @ X
        XtX_reg = XtX + lambda_ * np.eye(n_features, dtype=np.float64)

        try:
            L = linalg.cholesky(XtX_reg, lower=True)
            XtX_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
        except linalg.LinAlgError:
            XtX_inv = linalg.pinv(XtX_reg)

        T = XtX_inv @ X.T

    return np.ascontiguousarray(T)


def _process_batch_gpu(
    T_gpu,
    Y_batch: np.ndarray,
    n_rand: int,
    inv_perm_table: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Process a batch of cells using GPU.

    Args:
        T_gpu: Projection matrix on GPU (n_features, n_genes)
        Y_batch: Expression batch (n_genes, batch_size) - already transposed
        n_rand: Number of permutations
        inv_perm_table: Inverse permutation table

    Returns:
        Dict with beta, se, zscore, pvalue arrays
    """
    n_features = T_gpu.shape[0]
    batch_size = Y_batch.shape[1]

    # Transfer to GPU
    Y_gpu = cp.asarray(Y_batch, dtype=cp.float64)

    # Compute beta = T @ Y
    beta = T_gpu @ Y_gpu

    # Permutation testing
    aver = cp.zeros((n_features, batch_size), dtype=cp.float64)
    aver_sq = cp.zeros((n_features, batch_size), dtype=cp.float64)
    pvalue_counts = cp.zeros((n_features, batch_size), dtype=cp.float64)
    abs_beta = cp.abs(beta)

    # Process permutations
    for i in range(n_rand):
        inv_perm_idx = cp.asarray(inv_perm_table[i], dtype=cp.intp)
        T_perm = T_gpu[:, inv_perm_idx]
        beta_perm = T_perm @ Y_gpu

        pvalue_counts += (cp.abs(beta_perm) >= abs_beta).astype(cp.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

        del inv_perm_idx, T_perm, beta_perm

    # Finalize statistics
    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se = cp.sqrt(cp.maximum(var, 0.0))
    zscore = cp.where(se > EPS, (beta - mean) / se, 0.0)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)

    result = {
        'beta': cp.asnumpy(beta),
        'se': cp.asnumpy(se),
        'zscore': cp.asnumpy(zscore),
        'pvalue': cp.asnumpy(pvalue),
    }

    # Cleanup
    del Y_gpu, beta, aver, aver_sq, pvalue_counts, abs_beta
    del mean, var, se, zscore, pvalue
    cp.get_default_memory_pool().free_all_blocks()

    return result


def _process_batch_cpu(
    T: np.ndarray,
    Y_batch: np.ndarray,
    n_rand: int,
    inv_perm_table: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Process a batch of cells using CPU."""
    n_features = T.shape[0]
    batch_size = Y_batch.shape[1]

    # Compute beta = T @ Y
    beta = T @ Y_batch

    # Permutation testing
    aver = np.zeros((n_features, batch_size), dtype=np.float64)
    aver_sq = np.zeros((n_features, batch_size), dtype=np.float64)
    pvalue_counts = np.zeros((n_features, batch_size), dtype=np.float64)
    abs_beta = np.abs(beta)

    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]
        T_perm = T[:, inv_perm_idx]
        beta_perm = T_perm @ Y_batch

        pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se = np.sqrt(np.maximum(var, 0.0))
    zscore = np.where(se > EPS, (beta - mean) / se, 0.0)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)

    return {
        'beta': beta,
        'se': se,
        'zscore': zscore,
        'pvalue': pvalue,
    }


class SingleCellActivityInference:
    """
    GPU-optimized single-cell activity inference.

    Streams H5AD files and processes cells in batches for memory efficiency.

    Usage:
        >>> inference = SingleCellActivityInference(
        ...     signature_type="cytosig",
        ...     batch_size=10000,
        ...     use_gpu=True,
        ... )
        >>> inference.run(
        ...     h5ad_path="/path/to/data.h5ad",
        ...     output_path="/path/to/output.h5ad",
        ... )
    """

    def __init__(
        self,
        signature_type: Literal["cytosig", "secact"] = "cytosig",
        batch_size: int = 10000,
        lambda_: float = DEFAULT_LAMBDA,
        n_rand: int = DEFAULT_NRAND,
        seed: int = DEFAULT_SEED,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize single-cell activity inference.

        Args:
            signature_type: "cytosig" or "secact"
            batch_size: Number of cells per batch
            lambda_: Ridge regularization parameter
            n_rand: Number of permutations for significance testing
            seed: Random seed
            use_gpu: Use GPU if available
            verbose: Print progress messages
        """
        self.signature_type = signature_type
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.n_rand = n_rand
        self.seed = seed
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.verbose = verbose

        if self.use_gpu:
            self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check GPU availability."""
        try:
            n = cp.cuda.runtime.getDeviceCount()
            mem = cp.cuda.runtime.memGetInfo()
            if self.verbose:
                log(f"GPU: {n} device(s), {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free")
            return True
        except Exception as e:
            if self.verbose:
                log(f"GPU not available: {e}")
            self.use_gpu = False
            return False

    def _generate_inverse_perm_table(self, n_genes: int) -> np.ndarray:
        """Generate inverse permutation table for significance testing."""
        rng = np.random.default_rng(self.seed)
        inv_perm_table = np.zeros((self.n_rand, n_genes), dtype=np.int32)

        for i in range(self.n_rand):
            perm = rng.permutation(n_genes)
            inv_perm = np.argsort(perm)
            inv_perm_table[i] = inv_perm

        return inv_perm_table

    def run(
        self,
        h5ad_path: Union[str, Path],
        output_path: Union[str, Path],
        cell_subset: Optional[np.ndarray] = None,
        max_cells: Optional[int] = None,
        compression: str = "gzip",
    ) -> Path:
        """
        Run single-cell activity inference on H5AD file.

        Args:
            h5ad_path: Path to input H5AD file
            output_path: Path to output H5AD file
            cell_subset: Optional boolean mask or indices for cell subset
            max_cells: Maximum number of cells to process (for testing)
            compression: Output compression ("gzip" or None)

        Returns:
            Path to output H5AD file
        """
        h5ad_path = Path(h5ad_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            log("=" * 70)
            log(f"Single-Cell Activity Inference: {self.signature_type.upper()}")
            log("=" * 70)
            log(f"Input: {h5ad_path}")
            log(f"Output: {output_path}")
            log(f"Batch size: {self.batch_size:,}")
            log(f"Permutations: {self.n_rand}")
            log(f"Backend: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'}")

        start_time = time.time()

        # Open H5AD in backed mode for streaming
        if self.verbose:
            log("Opening H5AD in backed mode...")

        adata = ad.read_h5ad(h5ad_path, backed='r')
        n_cells, n_genes = adata.shape

        if self.verbose:
            log(f"  Shape: {n_cells:,} cells × {n_genes:,} genes")

        # Handle cell subsetting
        if cell_subset is not None:
            cell_indices = np.where(cell_subset)[0] if cell_subset.dtype == bool else cell_subset
        else:
            cell_indices = np.arange(n_cells)

        if max_cells is not None and len(cell_indices) > max_cells:
            cell_indices = cell_indices[:max_cells]

        n_cells_to_process = len(cell_indices)

        if self.verbose:
            log(f"  Processing: {n_cells_to_process:,} cells")

        # Load signature and match genes
        if self.verbose:
            log(f"Loading {self.signature_type} signature...")

        gene_names = list(adata.var_names)
        sig = load_signature_matrix(self.signature_type, genes=gene_names)

        if self.verbose:
            log(f"  Matched {sig.n_genes:,} genes × {sig.n_features} signatures")

        # Find gene indices in expression data
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        sig_gene_indices = [gene_to_idx[g] for g in sig.gene_names if g in gene_to_idx]

        if len(sig_gene_indices) < 100:
            raise ValueError(f"Too few matched genes: {len(sig_gene_indices)}")

        # Compute projection matrix
        if self.verbose:
            log("Computing projection matrix...")

        T = _compute_projection_matrix(sig.matrix, self.lambda_, self.use_gpu)

        # Generate inverse permutation table
        if self.verbose:
            log("Generating permutation table...")

        inv_perm_table = self._generate_inverse_perm_table(sig.n_genes)

        # Transfer T to GPU if using GPU
        T_gpu = None
        if self.use_gpu:
            T_gpu = cp.asarray(T, dtype=cp.float64)

        # Initialize output arrays
        n_batches = (n_cells_to_process + self.batch_size - 1) // self.batch_size

        # Process in batches and accumulate results
        all_zscore = []
        all_pvalue = []
        batch_times = []

        if self.verbose:
            log(f"Processing {n_batches} batches...")

        for batch_idx in range(n_batches):
            batch_start = time.time()

            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, n_cells_to_process)
            batch_cell_indices = cell_indices[start:end]

            # Load batch expression data
            X_batch = adata.X[batch_cell_indices][:, sig_gene_indices]

            if sparse.issparse(X_batch):
                X_dense = X_batch.toarray()
            else:
                X_dense = np.asarray(X_batch)

            # Transpose for matmul: (n_genes, batch_size)
            Y_batch = X_dense.T.astype(np.float64)

            # Process batch
            if self.use_gpu and T_gpu is not None:
                result = _process_batch_gpu(
                    T_gpu, Y_batch,
                    self.n_rand, inv_perm_table
                )
            else:
                result = _process_batch_cpu(
                    T, Y_batch,
                    self.n_rand, inv_perm_table
                )

            # Store results (transpose back to cells × features)
            all_zscore.append(result['zscore'].T)
            all_pvalue.append(result['pvalue'].T)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if self.verbose and (batch_idx % 10 == 0 or batch_idx == n_batches - 1):
                avg_time = np.mean(batch_times[-10:])
                eta = avg_time * (n_batches - batch_idx - 1)
                log(f"  Batch {batch_idx+1}/{n_batches} ({batch_time:.1f}s, ETA: {eta/60:.1f}min)")

            del X_batch, X_dense, Y_batch, result
            gc.collect()

        # Cleanup GPU
        if T_gpu is not None:
            del T_gpu
            cp.get_default_memory_pool().free_all_blocks()

        # Concatenate results
        if self.verbose:
            log("Concatenating results...")

        zscore_matrix = np.vstack(all_zscore)
        pvalue_matrix = np.vstack(all_pvalue)

        # Create output AnnData
        if self.verbose:
            log("Creating output AnnData...")

        # Get cell metadata
        cell_obs = adata.obs.iloc[cell_indices].copy()

        output_adata = ad.AnnData(
            X=zscore_matrix.astype(np.float32),
            obs=cell_obs,
            var=pd.DataFrame(
                index=sig.feature_names,
                data={'signature': sig.feature_names}
            ),
        )
        output_adata.layers['pvalue'] = pvalue_matrix.astype(np.float32)
        output_adata.uns['signature'] = self.signature_type
        output_adata.uns['lambda'] = self.lambda_
        output_adata.uns['n_rand'] = self.n_rand
        output_adata.uns['n_matched_genes'] = len(sig_gene_indices)

        # Write output
        if self.verbose:
            log(f"Writing to {output_path}...")

        output_adata.write_h5ad(output_path, compression=compression)

        total_time = time.time() - start_time

        if self.verbose:
            log("")
            log(f"Completed in {total_time/60:.1f} min")
            log(f"  Processed: {n_cells_to_process:,} cells")
            log(f"  Output: {output_path.name} ({output_path.stat().st_size / 1024**2:.1f} MB)")

        return output_path


def run_singlecell_activity(
    h5ad_path: Union[str, Path],
    output_dir: Union[str, Path],
    signature_types: List[str] = ["cytosig", "secact"],
    batch_size: int = 10000,
    n_rand: int = 1000,
    use_gpu: bool = True,
    max_cells: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Run single-cell activity inference for multiple signatures.

    Args:
        h5ad_path: Path to input H5AD file
        output_dir: Output directory
        signature_types: List of signatures to run
        batch_size: Cells per batch
        n_rand: Number of permutations
        use_gpu: Use GPU if available
        max_cells: Maximum cells (for testing)
        verbose: Print progress

    Returns:
        Dict mapping signature types to output paths
    """
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    for sig_type in signature_types:
        output_path = output_dir / f"{h5ad_path.stem}_{sig_type}.h5ad"

        inference = SingleCellActivityInference(
            signature_type=sig_type,
            batch_size=batch_size,
            n_rand=n_rand,
            use_gpu=use_gpu,
            verbose=verbose,
        )

        output_paths[sig_type] = inference.run(
            h5ad_path=h5ad_path,
            output_path=output_path,
            max_cells=max_cells,
        )

    return output_paths


if __name__ == "__main__":
    print("=" * 60)
    print("Single-Cell Activity Inference Module - Testing")
    print("=" * 60)

    # Quick test with synthetic data
    import tempfile

    n_cells = 1000
    n_genes = 5000

    print(f"\nTest data: {n_cells} cells × {n_genes} genes")

    # Create synthetic AnnData
    np.random.seed(42)
    X = np.random.poisson(1.5, (n_cells, n_genes)).astype(np.float32)
    X = sparse.csr_matrix(X)

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    cell_ids = [f"CELL{i}" for i in range(n_cells)]

    # Add some real gene names for signature matching
    from secactpy import load_cytosig
    cytosig = load_cytosig()
    real_genes = list(cytosig.index)[:min(1000, n_genes)]
    gene_names[:len(real_genes)] = real_genes

    test_adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_ids, data={'cell_type': np.random.choice(['A', 'B', 'C'], n_cells)}),
        var=pd.DataFrame(index=gene_names),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test_input.h5ad"
        output_path = Path(tmpdir) / "test_output.h5ad"

        test_adata.write_h5ad(input_path)

        print(f"\nRunning single-cell activity inference...")

        inference = SingleCellActivityInference(
            signature_type="cytosig",
            batch_size=200,
            n_rand=100,
            use_gpu=CUPY_AVAILABLE,
            verbose=True,
        )

        result_path = inference.run(
            h5ad_path=input_path,
            output_path=output_path,
            max_cells=500,
        )

        # Load and check results
        result = ad.read_h5ad(result_path)
        print(f"\nResult shape: {result.shape}")
        print(f"Signatures: {result.var_names[:5].tolist()}")
        print(f"Z-score range: [{result.X.min():.2f}, {result.X.max():.2f}]")

    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
