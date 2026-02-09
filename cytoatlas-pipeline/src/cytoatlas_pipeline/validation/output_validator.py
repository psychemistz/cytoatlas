"""
Output validation for pipeline results.

Validates that generated outputs meet quality standards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, name: str, passed: bool, message: str, details: dict | None = None):
        """
        Initialize validation result.

        Args:
            name: Validation check name
            passed: Whether validation passed
            message: Result message
            details: Additional details
        """
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class OutputValidator:
    """
    Validator for pipeline output files.

    Performs quality checks on generated JSON and CSV files.

    Example:
        >>> validator = OutputValidator()
        >>> result = validator.validate_activity_range(data)
        >>> all_results = validator.run_all("results/cima/")
    """

    def __init__(self):
        """Initialize output validator."""
        pass

    def validate_activity_range(
        self,
        data: list[dict],
        field: str = "activity",
        min_val: float = -10.0,
        max_val: float = 10.0,
    ) -> ValidationResult:
        """
        Validate that activity z-scores are in expected range.

        Args:
            data: List of data records
            field: Field name to check (default: 'activity')
            min_val: Minimum expected value
            max_val: Maximum expected value

        Returns:
            ValidationResult
        """
        if not data:
            return ValidationResult(
                "activity_range",
                False,
                "No data provided",
            )

        values = []
        for record in data:
            if field in record and record[field] is not None:
                try:
                    values.append(float(record[field]))
                except (ValueError, TypeError):
                    pass

        if not values:
            return ValidationResult(
                "activity_range",
                False,
                f"No valid values found for field '{field}'",
            )

        values_arr = np.array(values)
        actual_min = values_arr.min()
        actual_max = values_arr.max()
        actual_mean = values_arr.mean()
        actual_std = values_arr.std()

        # Check if values are within expected range
        out_of_range = ((values_arr < min_val) | (values_arr > max_val)).sum()
        out_of_range_pct = out_of_range / len(values) * 100

        passed = out_of_range_pct < 1.0  # Allow <1% outliers

        message = (
            f"Activity range check: {actual_min:.2f} to {actual_max:.2f} "
            f"(mean: {actual_mean:.2f}, std: {actual_std:.2f}). "
            f"Out of range: {out_of_range_pct:.1f}%"
        )

        return ValidationResult(
            "activity_range",
            passed,
            message,
            details={
                "min": float(actual_min),
                "max": float(actual_max),
                "mean": float(actual_mean),
                "std": float(actual_std),
                "out_of_range_count": int(out_of_range),
                "out_of_range_pct": float(out_of_range_pct),
                "total_values": len(values),
            },
        )

    def validate_gene_overlap(
        self,
        data: list[dict],
        signatures: dict[str, list[str]],
        min_overlap: float = 0.8,
    ) -> ValidationResult:
        """
        Validate gene overlap between data and signatures.

        Args:
            data: List of data records
            signatures: Dictionary mapping signature names to gene lists
            min_overlap: Minimum required overlap (default: 0.8 = 80%)

        Returns:
            ValidationResult
        """
        if not data or not signatures:
            return ValidationResult(
                "gene_overlap",
                False,
                "No data or signatures provided",
            )

        # Extract genes from data
        data_genes = set()
        for record in data:
            if "gene" in record:
                data_genes.add(record["gene"])

        if not data_genes:
            return ValidationResult(
                "gene_overlap",
                False,
                "No genes found in data",
            )

        # Check overlap for each signature
        overlaps = {}
        for sig_name, sig_genes in signatures.items():
            sig_genes_set = set(sig_genes)
            overlap = data_genes & sig_genes_set
            overlap_pct = len(overlap) / len(sig_genes_set) * 100 if sig_genes_set else 0
            overlaps[sig_name] = overlap_pct

        avg_overlap = np.mean(list(overlaps.values())) if overlaps else 0
        passed = avg_overlap >= (min_overlap * 100)

        message = f"Gene overlap: {avg_overlap:.1f}% average across {len(overlaps)} signatures"

        return ValidationResult(
            "gene_overlap",
            passed,
            message,
            details={
                "avg_overlap_pct": float(avg_overlap),
                "n_data_genes": len(data_genes),
                "n_signatures": len(signatures),
                "signature_overlaps": {k: float(v) for k, v in overlaps.items()},
            },
        )

    def validate_field_completeness(
        self,
        data: list[dict],
        required_fields: list[str],
    ) -> ValidationResult:
        """
        Validate that required fields are present in all records.

        Args:
            data: List of data records
            required_fields: List of required field names

        Returns:
            ValidationResult
        """
        if not data:
            return ValidationResult(
                "field_completeness",
                False,
                "No data provided",
            )

        # Check each field
        field_stats = {}
        for field in required_fields:
            present = sum(1 for record in data if field in record and record[field] is not None)
            pct = present / len(data) * 100
            field_stats[field] = {
                "present": present,
                "total": len(data),
                "pct": pct,
            }

        # Check if all fields are >95% complete
        incomplete_fields = [
            field for field, stats in field_stats.items() if stats["pct"] < 95.0
        ]

        passed = len(incomplete_fields) == 0

        if passed:
            message = f"All {len(required_fields)} required fields are >95% complete"
        else:
            message = f"Incomplete fields: {incomplete_fields}"

        return ValidationResult(
            "field_completeness",
            passed,
            message,
            details=field_stats,
        )

    def validate_schema(
        self,
        data: list[dict],
        expected_schema: dict[str, type],
    ) -> ValidationResult:
        """
        Validate data structure matches expected schema.

        Args:
            data: List of data records
            expected_schema: Dictionary mapping field names to expected types

        Returns:
            ValidationResult
        """
        if not data:
            return ValidationResult(
                "schema",
                False,
                "No data provided",
            )

        # Sample first record for schema check
        sample = data[0]
        errors = []

        for field, expected_type in expected_schema.items():
            if field not in sample:
                errors.append(f"Missing field: {field}")
            elif not isinstance(sample[field], expected_type):
                actual_type = type(sample[field]).__name__
                expected_name = expected_type.__name__
                errors.append(f"Field '{field}': expected {expected_name}, got {actual_type}")

        passed = len(errors) == 0

        if passed:
            message = f"Schema validation passed for {len(expected_schema)} fields"
        else:
            message = f"Schema validation failed: {len(errors)} errors"

        return ValidationResult(
            "schema",
            passed,
            message,
            details={"errors": errors},
        )

    def run_all(
        self,
        output_dir: Path | str,
        save_report: bool = True,
    ) -> dict[str, list[ValidationResult]]:
        """
        Run all validations on an output directory.

        Args:
            output_dir: Directory containing output JSON files
            save_report: Whether to save validation report

        Returns:
            Dictionary mapping file names to validation results
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return {}

        results = {}

        # Find all JSON files
        json_files = list(output_dir.glob("**/*.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Skip if not a list
                if not isinstance(data, list):
                    continue

                file_results = []

                # Run activity range check if 'activity' field exists
                if data and "activity" in data[0]:
                    file_results.append(self.validate_activity_range(data))

                # Run field completeness check
                if data:
                    fields = list(data[0].keys())
                    file_results.append(
                        self.validate_field_completeness(data, fields)
                    )

                results[str(json_file.relative_to(output_dir))] = file_results

            except Exception as e:
                logger.warning(f"Failed to validate {json_file}: {e}")

        # Save report if requested
        if save_report:
            report_path = output_dir / ".validation_report.json"
            report = {
                "output_dir": str(output_dir),
                "n_files_validated": len(results),
                "results": {
                    file: [r.to_dict() for r in file_results]
                    for file, file_results in results.items()
                },
            }

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Validation report saved to {report_path}")

        return results
