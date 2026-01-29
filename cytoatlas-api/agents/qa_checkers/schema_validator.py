"""
Schema Validator Agent.

Validates API responses against Pydantic schemas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""

    schema_name: str
    is_valid: bool
    errors: List[str]
    sample_data: Optional[Dict[str, Any]] = None


class SchemaValidator:
    """Validates data against Pydantic schemas."""

    def __init__(self, schemas_path: Optional[Path] = None):
        self.schemas_path = schemas_path or Path(__file__).parent.parent.parent / "app" / "schemas"
        self._loaded_schemas: Dict[str, Type[BaseModel]] = {}

    def load_schema(self, module_name: str, schema_name: str) -> Optional[Type[BaseModel]]:
        """Dynamically load a Pydantic schema."""
        cache_key = f"{module_name}.{schema_name}"
        if cache_key in self._loaded_schemas:
            return self._loaded_schemas[cache_key]

        try:
            import importlib

            module = importlib.import_module(f"app.schemas.{module_name}")
            schema = getattr(module, schema_name, None)
            if schema and issubclass(schema, BaseModel):
                self._loaded_schemas[cache_key] = schema
                return schema
        except (ImportError, AttributeError):
            pass

        return None

    def validate(
        self,
        data: Any,
        module_name: str,
        schema_name: str,
    ) -> SchemaValidationResult:
        """Validate data against a schema."""
        schema = self.load_schema(module_name, schema_name)

        if schema is None:
            return SchemaValidationResult(
                schema_name=schema_name,
                is_valid=False,
                errors=[f"Schema '{module_name}.{schema_name}' not found"],
            )

        errors: List[str] = []

        # Handle list of items
        if isinstance(data, list):
            for i, item in enumerate(data[:5]):  # Check first 5 items
                try:
                    schema.model_validate(item)
                except ValidationError as e:
                    for error in e.errors():
                        errors.append(f"Item {i}: {error['loc']} - {error['msg']}")
        else:
            try:
                schema.model_validate(data)
            except ValidationError as e:
                for error in e.errors():
                    errors.append(f"{error['loc']} - {error['msg']}")

        return SchemaValidationResult(
            schema_name=schema_name,
            is_valid=len(errors) == 0,
            errors=errors,
            sample_data=data if isinstance(data, dict) else (data[0] if data else None),
        )

    def validate_json_file(
        self,
        file_path: Path,
        module_name: str,
        schema_name: str,
    ) -> SchemaValidationResult:
        """Validate a JSON file against a schema."""
        try:
            with open(file_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return SchemaValidationResult(
                schema_name=schema_name,
                is_valid=False,
                errors=[f"Failed to load JSON: {e}"],
            )

        return self.validate(data, module_name, schema_name)

    def validate_all_cima_schemas(self, data_dir: Path) -> Dict[str, SchemaValidationResult]:
        """Validate all CIMA data files against their schemas."""
        results = {}

        schema_mappings = {
            "cima_correlations.json": ("cima", "CIMACorrelation"),
            "cima_celltype.json": ("cima", "CIMACellTypeActivity"),
            "cima_differential.json": ("cima", "CIMADifferentialResult"),
            "cima_eqtl_top.json": ("cima", "CIMAeQTLResult"),
        }

        for filename, (module, schema) in schema_mappings.items():
            file_path = data_dir / filename
            if file_path.exists():
                results[filename] = self.validate_json_file(file_path, module, schema)
            else:
                results[filename] = SchemaValidationResult(
                    schema_name=schema,
                    is_valid=False,
                    errors=[f"File not found: {filename}"],
                )

        return results

    def validate_validation_schemas(self, data_dir: Path) -> Dict[str, SchemaValidationResult]:
        """Validate validation data files against schemas."""
        results = {}

        schema_mappings = {
            "sample_validation.json": ("validation", "SampleLevelValidation"),
            "celltype_validation.json": ("validation", "CellTypeLevelValidation"),
            "singlecell_validation.json": ("validation", "SingleCellDirectValidation"),
            "biological_associations.json": ("validation", "BiologicalValidationTable"),
            "gene_coverage.json": ("validation", "GeneCoverage"),
            "cv_stability.json": ("validation", "CVStability"),
        }

        for filename, (module, schema) in schema_mappings.items():
            file_path = data_dir / filename
            if file_path.exists():
                results[filename] = self.validate_json_file(file_path, module, schema)
            else:
                results[filename] = SchemaValidationResult(
                    schema_name=schema,
                    is_valid=False,
                    errors=[f"File not found: {filename}"],
                )

        return results

    def report(self, results: Dict[str, SchemaValidationResult]) -> dict[str, Any]:
        """Generate validation report."""
        total = len(results)
        valid = sum(1 for r in results.values() if r.is_valid)

        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "pass_rate": f"{valid / total * 100:.1f}%" if total > 0 else "N/A",
            "details": {
                filename: {
                    "schema": r.schema_name,
                    "valid": r.is_valid,
                    "errors": r.errors,
                }
                for filename, r in results.items()
            },
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate CytoAtlas data schemas")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing JSON data files",
    )
    parser.add_argument(
        "--type",
        choices=["cima", "validation", "all"],
        default="all",
        help="Type of schemas to validate",
    )

    args = parser.parse_args()

    validator = SchemaValidator()

    if args.type == "cima":
        results = validator.validate_all_cima_schemas(args.data_dir)
    elif args.type == "validation":
        results = validator.validate_validation_schemas(args.data_dir)
    else:
        results = {
            **validator.validate_all_cima_schemas(args.data_dir),
            **validator.validate_validation_schemas(args.data_dir),
        }

    report = validator.report(results)
    print(json.dumps(report, indent=2))
