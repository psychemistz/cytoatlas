"""
GPU resource management.

Provides:
- GPU device selection and memory management
- CuPy/NumPy backend abstraction
- Multi-GPU coordination
- Memory pool management
"""

from __future__ import annotations

import gc
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    import cupy as cp

# Global GPU state
_gpu_manager: Optional["GPUManager"] = None


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    device_id: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: tuple[int, int]


def check_cupy_available() -> tuple[bool, Optional[str]]:
    """
    Check if CuPy is available and functional.

    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        import cupy as cp

        # Test GPU availability
        _ = cp.array([1.0])
        cp.cuda.Device().synchronize()
        return True, None
    except ImportError:
        return False, "CuPy not installed. Install with: pip install cupy-cuda12x"
    except Exception as e:
        return False, f"GPU initialization failed: {e}"


class GPUManager:
    """
    Manages GPU resources for pipeline operations.

    Handles device selection, memory pooling, and provides a unified
    interface for GPU/CPU operations.

    Example:
        >>> manager = GPUManager(devices=[0, 1])
        >>> with manager.device(0):
        ...     arr_gpu = manager.to_gpu(arr_cpu)
        ...     result_gpu = arr_gpu @ arr_gpu.T
        ...     result_cpu = manager.to_cpu(result_gpu)
    """

    def __init__(
        self,
        devices: list[int] | None = None,
        memory_fraction: float = 0.9,
        allow_growth: bool = True,
        fallback_to_cpu: bool = True,
    ):
        """
        Initialize GPU manager.

        Args:
            devices: List of GPU device IDs to use. None for auto-detect.
            memory_fraction: Fraction of GPU memory to use (0-1).
            allow_growth: Allow memory to grow dynamically.
            fallback_to_cpu: Fall back to CPU if GPU unavailable.
        """
        self.memory_fraction = memory_fraction
        self.allow_growth = allow_growth
        self.fallback_to_cpu = fallback_to_cpu

        # Check CuPy availability
        self._cupy_available, self._cupy_error = check_cupy_available()

        if self._cupy_available:
            import cupy as cp

            self._cp = cp

            # Get available devices
            n_devices = cp.cuda.runtime.getDeviceCount()
            if devices is None:
                devices = list(range(n_devices))
            else:
                # Validate device IDs
                invalid = [d for d in devices if d >= n_devices]
                if invalid:
                    raise ValueError(
                        f"Invalid GPU device IDs: {invalid}. "
                        f"Available: {list(range(n_devices))}"
                    )

            self.devices = devices
            self._current_device = devices[0] if devices else 0

            # Initialize memory pools
            self._init_memory_pools()
        else:
            self._cp = None
            self.devices = []
            self._current_device = -1

            if not fallback_to_cpu:
                raise RuntimeError(f"GPU required but not available: {self._cupy_error}")

    def _init_memory_pools(self) -> None:
        """Initialize CuPy memory pools for each device."""
        if not self._cupy_available:
            return

        for device_id in self.devices:
            with self._cp.cuda.Device(device_id):
                # Get memory pool
                mempool = self._cp.get_default_memory_pool()

                # Set memory limit if not using growth mode
                if not self.allow_growth:
                    props = self._cp.cuda.runtime.getDeviceProperties(device_id)
                    total_memory = props["totalGlobalMem"]
                    limit = int(total_memory * self.memory_fraction)
                    mempool.set_limit(size=limit)

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self._cupy_available and len(self.devices) > 0

    @property
    def backend(self) -> Literal["cupy", "numpy"]:
        """Get current backend name."""
        return "cupy" if self.is_gpu_available else "numpy"

    @property
    def xp(self) -> Any:
        """Get array module (cupy or numpy)."""
        if self.is_gpu_available:
            return self._cp
        return np

    def get_device_info(self, device_id: int | None = None) -> GPUInfo | None:
        """Get information about a GPU device."""
        if not self.is_gpu_available:
            return None

        if device_id is None:
            device_id = self._current_device

        with self._cp.cuda.Device(device_id):
            props = self._cp.cuda.runtime.getDeviceProperties(device_id)
            meminfo = self._cp.cuda.runtime.memGetInfo()

            return GPUInfo(
                device_id=device_id,
                name=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                total_memory_gb=props["totalGlobalMem"] / (1024**3),
                free_memory_gb=meminfo[0] / (1024**3),
                compute_capability=(props["major"], props["minor"]),
            )

    def get_all_device_info(self) -> list[GPUInfo]:
        """Get information about all managed GPU devices."""
        if not self.is_gpu_available:
            return []
        return [self.get_device_info(d) for d in self.devices]

    @contextmanager
    def device(self, device_id: int | None = None) -> Generator[None, None, None]:
        """
        Context manager for using a specific GPU device.

        Args:
            device_id: GPU device ID. None uses current device.

        Yields:
            None
        """
        if not self.is_gpu_available:
            yield
            return

        if device_id is None:
            device_id = self._current_device

        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not in managed devices: {self.devices}")

        old_device = self._current_device
        self._current_device = device_id

        with self._cp.cuda.Device(device_id):
            try:
                yield
            finally:
                self._current_device = old_device

    def to_gpu(self, arr: np.ndarray, device_id: int | None = None) -> Any:
        """
        Transfer array to GPU.

        Args:
            arr: NumPy array to transfer.
            device_id: Target device ID. None uses current device.

        Returns:
            CuPy array on GPU, or original array if GPU unavailable.
        """
        if not self.is_gpu_available:
            return arr

        if device_id is None:
            device_id = self._current_device

        with self._cp.cuda.Device(device_id):
            return self._cp.asarray(arr)

    def to_cpu(self, arr: Any) -> np.ndarray:
        """
        Transfer array to CPU.

        Args:
            arr: Array to transfer (CuPy or NumPy).

        Returns:
            NumPy array on CPU.
        """
        if not self.is_gpu_available:
            return np.asarray(arr)

        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def free_memory(self, device_id: int | None = None) -> None:
        """
        Free GPU memory on a device.

        Args:
            device_id: Device to free memory on. None frees all devices.
        """
        if not self.is_gpu_available:
            return

        devices = [device_id] if device_id is not None else self.devices

        for d in devices:
            with self._cp.cuda.Device(d):
                self._cp.get_default_memory_pool().free_all_blocks()
                self._cp.get_default_pinned_memory_pool().free_all_blocks()

        gc.collect()

    def synchronize(self, device_id: int | None = None) -> None:
        """
        Synchronize GPU operations.

        Args:
            device_id: Device to synchronize. None synchronizes current device.
        """
        if not self.is_gpu_available:
            return

        if device_id is None:
            device_id = self._current_device

        with self._cp.cuda.Device(device_id):
            self._cp.cuda.Device(device_id).synchronize()

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | str = np.float64,
        device_id: int | None = None,
    ) -> Any:
        """
        Allocate array on GPU or CPU.

        Args:
            shape: Array shape.
            dtype: Data type.
            device_id: Target device ID. None uses current device.

        Returns:
            Allocated array (CuPy on GPU, NumPy on CPU).
        """
        if not self.is_gpu_available:
            return np.zeros(shape, dtype=dtype)

        if device_id is None:
            device_id = self._current_device

        with self._cp.cuda.Device(device_id):
            return self._cp.zeros(shape, dtype=dtype)

    def zeros_like(self, arr: Any) -> Any:
        """Create zero array with same shape and type as input."""
        if self.is_gpu_available and hasattr(arr, "device"):
            return self._cp.zeros_like(arr)
        return np.zeros_like(arr)

    def ones_like(self, arr: Any) -> Any:
        """Create ones array with same shape and type as input."""
        if self.is_gpu_available and hasattr(arr, "device"):
            return self._cp.ones_like(arr)
        return np.ones_like(arr)


def get_gpu_manager() -> GPUManager:
    """
    Get the global GPU manager instance.

    Creates a default manager on first call.

    Returns:
        Global GPUManager instance.
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def set_gpu_manager(manager: GPUManager) -> None:
    """
    Set the global GPU manager instance.

    Args:
        manager: GPUManager instance to use globally.
    """
    global _gpu_manager
    _gpu_manager = manager


def reset_gpu_manager() -> None:
    """Reset the global GPU manager."""
    global _gpu_manager
    if _gpu_manager is not None:
        _gpu_manager.free_memory()
    _gpu_manager = None
