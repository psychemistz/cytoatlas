"""Data version tracking for cache invalidation."""

import hashlib
import time
from pathlib import Path
from typing import Any


class DataVersionTracker:
    """
    Track file versions via checksums and modification times.

    Used to automatically invalidate cache when data files change.
    """

    _instance: "DataVersionTracker | None" = None
    _versions: dict[str, dict[str, Any]] = {}

    def __new__(cls) -> "DataVersionTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._versions = {}
        return cls._instance

    def get_version(self, filepath: Path | str) -> dict[str, Any]:
        """
        Get current version info for a file.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with checksum, mtime, size
        """
        path = Path(filepath)

        if not path.exists():
            return {
                "checksum": None,
                "mtime": 0,
                "size": 0,
            }

        stat = path.stat()

        # Compute MD5 checksum for small files (<100MB)
        # For large files, use mtime + size as proxy
        if stat.st_size < 100 * 1024 * 1024:
            checksum = self._compute_md5(path)
        else:
            # Use mtime + size as pseudo-checksum for large files
            checksum = f"mtime:{stat.st_mtime}:size:{stat.st_size}"

        return {
            "checksum": checksum,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }

    def has_changed(self, filepath: Path | str) -> bool:
        """
        Check if file has changed since last check.

        Args:
            filepath: Path to file

        Returns:
            True if file changed or is new, False otherwise
        """
        path_str = str(filepath)

        # Get current version
        current_version = self.get_version(filepath)

        # Get stored version
        stored_version = self._versions.get(path_str)

        # First time seeing this file
        if stored_version is None:
            self._versions[path_str] = current_version
            return True

        # Compare checksums
        changed = current_version["checksum"] != stored_version["checksum"]

        # Update stored version if changed
        if changed:
            self._versions[path_str] = current_version

        return changed

    def mark_checked(self, filepath: Path | str) -> None:
        """
        Mark file as checked (update stored version).

        Args:
            filepath: Path to file
        """
        path_str = str(filepath)
        self._versions[path_str] = self.get_version(filepath)

    def invalidate(self, filepath: Path | str) -> None:
        """
        Invalidate a file's stored version.

        Args:
            filepath: Path to file
        """
        path_str = str(filepath)
        if path_str in self._versions:
            del self._versions[path_str]

    def clear(self) -> None:
        """Clear all version tracking."""
        self._versions.clear()

    def _compute_md5(self, filepath: Path) -> str:
        """
        Compute MD5 checksum of file.

        Args:
            filepath: Path to file

        Returns:
            MD5 hex digest
        """
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def get_stats(self) -> dict[str, Any]:
        """Get tracking statistics."""
        return {
            "tracked_files": len(self._versions),
            "files": list(self._versions.keys()),
        }


def get_version_tracker() -> DataVersionTracker:
    """Get global version tracker instance."""
    return DataVersionTracker()
