#!/usr/bin/env python3
"""
CIMA Panel Validator Agent

Validates all CIMA visualization panels and reports issues.
Can trigger the panel fixer agent when problems are detected.

Usage:
    python cima_panel_validator.py [--fix] [--verbose] [--panel PANEL_NAME]
"""

import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

# Paths
VIZ_DATA_DIR = Path('/vf/users/parks34/projects/2cytoatlas/visualization/data')
EMBEDDED_DATA_PATH = VIZ_DATA_DIR / 'embedded_data.js'
LOG_DIR = Path('/vf/users/parks34/projects/2cytoatlas/logs')
LOG_DIR.mkdir(exist_ok=True)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    panel: str
    check_name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    fix_hint: Optional[str] = None


@dataclass
class PanelStatus:
    """Overall status of a panel."""
    name: str
    functional: bool
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]


class CIMAPanelValidator:
    """Validates CIMA visualization panels."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.data: Dict[str, Any] = {}
        self.panel_statuses: Dict[str, PanelStatus] = {}

    def load_data(self) -> bool:
        """Load embedded data or individual JSON files."""
        try:
            # Try embedded data first
            if EMBEDDED_DATA_PATH.exists():
                content = EMBEDDED_DATA_PATH.read_text()
                # Remove JS wrapper
                content = content.replace('const EMBEDDED_DATA = ', '').rstrip(';')
                self.data = json.loads(content)
                if self.verbose:
                    print(f"Loaded embedded data with {len(self.data)} keys")
                return True
        except Exception as e:
            if self.verbose:
                print(f"Could not load embedded data: {e}")

        # Fallback to individual JSON files
        try:
            for json_file in VIZ_DATA_DIR.glob('*.json'):
                key = json_file.stem.replace('_', '').replace('-', '')
                with open(json_file) as f:
                    self.data[key] = json.load(f)
            if self.verbose:
                print(f"Loaded {len(self.data)} individual JSON files")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _check(self, panel: str, check_name: str, condition: bool,
               message: str, severity: str = "error", fix_hint: str = None) -> ValidationResult:
        """Helper to create validation result."""
        result = ValidationResult(
            panel=panel,
            check_name=check_name,
            passed=condition,
            message=message if not condition else f"OK: {check_name}",
            severity=severity,
            fix_hint=fix_hint
        )
        if self.verbose or not condition:
            status = "✓" if condition else ("⚠" if severity == "warning" else "✗")
            print(f"  {status} [{panel}] {check_name}: {message}")
        return result

    # =========================================================================
    # Panel Validators
    # =========================================================================

    def validate_celltype_panel(self) -> PanelStatus:
        """Validate Cell Type Overview panel."""
        panel = "Cell Type"
        results = []
        data = self.data.get('cimacelltype')

        # Check data exists
        results.append(self._check(
            panel, "data_exists", bool(data),
            "cimacelltype data missing" if not data else "Data loaded",
            fix_hint="Run: preprocess_cima_celltype()"
        ))

        if data:
            # Data can be list of records or dict with keys
            if isinstance(data, list):
                # List format: [{cell_type, signature, mean_activity, ...}, ...]
                results.append(self._check(
                    panel, "record_count", len(data) >= 100,
                    f"Only {len(data)} records (expected ≥100)"
                ))

                if data:
                    # Check required fields in records
                    first = data[0]
                    required_fields = ['cell_type', 'signature', 'mean_activity']
                    for field in required_fields:
                        results.append(self._check(
                            panel, f"has_{field}", field in first,
                            f"Records missing '{field}' field"
                        ))

                    # Count unique cell types and signatures
                    cell_types = set(d.get('cell_type') for d in data if d.get('cell_type'))
                    signatures = set(d.get('signature') for d in data if d.get('signature'))

                    results.append(self._check(
                        panel, "cell_types_count", len(cell_types) >= 10,
                        f"Only {len(cell_types)} unique cell types (expected ≥10)",
                        severity="warning"
                    ))

                    results.append(self._check(
                        panel, "signatures_count", len(signatures) >= 20,
                        f"Only {len(signatures)} unique signatures (expected ≥20)",
                        severity="warning"
                    ))
            else:
                # Dict format with cell_types, signatures, heatmap_data keys
                for key in ['cell_types', 'signatures', 'heatmap_data']:
                    results.append(self._check(
                        panel, f"has_{key}", key in data,
                        f"Missing '{key}' in data"
                    ))

                cell_types = data.get('cell_types', [])
                results.append(self._check(
                    panel, "cell_types_count", len(cell_types) >= 10,
                    f"Only {len(cell_types)} cell types (expected ≥10)",
                    severity="warning"
                ))

                signatures = data.get('signatures', [])
                results.append(self._check(
                    panel, "signatures_count", len(signatures) >= 20,
                    f"Only {len(signatures)} signatures (expected ≥20)",
                    severity="warning"
                ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_biochem_scatter_panel(self) -> PanelStatus:
        """Validate Biochemistry Scatter panel."""
        panel = "Biochem Scatter"
        results = []
        data = self.data.get('cimabiochemscatter', {})

        # Check data exists
        results.append(self._check(
            panel, "data_exists", bool(data),
            "cimabiochemscatter data missing" if not data else "Data loaded",
            fix_hint="Run: preprocess_cima_biochem_scatter()"
        ))

        if data:
            # Check required keys
            for key in ['samples', 'biochem_features', 'cytokines']:
                results.append(self._check(
                    panel, f"has_{key}", key in data,
                    f"Missing '{key}' in data"
                ))

            # Check SecAct proteins (new requirement)
            results.append(self._check(
                panel, "has_secact_proteins", 'secact_proteins' in data,
                "Missing 'secact_proteins' - SecAct dropdown won't work",
                fix_hint="Update preprocess_cima_biochem_scatter() to include SecAct"
            ))

            # Check sample count
            samples = data.get('samples', [])
            results.append(self._check(
                panel, "sample_count", len(samples) >= 100,
                f"Only {len(samples)} samples (expected ≥100)"
            ))

            # Check sex annotation
            if samples:
                sex_valid = sum(1 for s in samples if s.get('sex') in ['Male', 'Female'])
                sex_pct = sex_valid / len(samples) * 100
                results.append(self._check(
                    panel, "sex_annotation", sex_pct > 80,
                    f"Only {sex_pct:.1f}% samples have valid sex annotation",
                    fix_hint="Fix sex field extraction in preprocessing"
                ))

                # Check activity data
                has_activity = sum(1 for s in samples if s.get('activity'))
                results.append(self._check(
                    panel, "activity_data", has_activity == len(samples),
                    f"{len(samples) - has_activity} samples missing CytoSig activity"
                ))

                # Check SecAct activity data
                has_secact = sum(1 for s in samples if s.get('secact_activity'))
                results.append(self._check(
                    panel, "secact_activity_data", has_secact > 0,
                    f"No samples have SecAct activity data",
                    severity="warning" if has_secact == 0 else "info"
                ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_population_panel(self) -> PanelStatus:
        """Validate Population Stratification panel."""
        panel = "Population"
        results = []
        data = self.data.get('cimapopulationstratification', {})

        # Check data exists
        results.append(self._check(
            panel, "data_exists", bool(data),
            "cimapopulationstratification data missing" if not data else "Data loaded",
            fix_hint="Run: preprocess_cima_population_stratification()"
        ))

        if data:
            # Check required keys
            for key in ['cytokines', 'groups', 'effect_sizes']:
                results.append(self._check(
                    panel, f"has_{key}", key in data,
                    f"Missing '{key}' in data"
                ))

            # Check all stratifications exist
            required_strats = ['sex', 'age', 'bmi', 'blood_type', 'smoking']
            groups = data.get('groups', {})
            effect_sizes = data.get('effect_sizes', {})

            for strat in required_strats:
                # Check groups
                results.append(self._check(
                    panel, f"groups_{strat}", strat in groups and len(groups.get(strat, {})) > 0,
                    f"Missing or empty groups for '{strat}'",
                    fix_hint=f"Add {strat} stratification to preprocessing"
                ))

                # Check effect sizes
                effects = effect_sizes.get(strat, [])
                results.append(self._check(
                    panel, f"effects_{strat}", len(effects) >= 20,
                    f"Only {len(effects)} effect sizes for '{strat}' (expected ≥20)",
                    severity="warning" if len(effects) > 0 else "error"
                ))

            # Check cytokines count
            cytokines = data.get('cytokines', [])
            results.append(self._check(
                panel, "cytokines_count", len(cytokines) >= 40,
                f"Only {len(cytokines)} cytokines (expected ≥40 for CytoSig)",
                severity="warning"
            ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_multiomics_panel(self) -> PanelStatus:
        """Validate Multi-omics Integration panel."""
        panel = "Multi-omics"
        results = []

        # Check correlations data
        corr_data = self.data.get('cimacorrelations', {})
        results.append(self._check(
            panel, "correlations_exists", bool(corr_data),
            "cimacorrelations data missing",
            fix_hint="Run: preprocess_cima_correlations()"
        ))

        if corr_data:
            for key in ['age', 'bmi', 'biochemistry']:
                items = corr_data.get(key, [])
                results.append(self._check(
                    panel, f"corr_{key}", len(items) > 0,
                    f"No {key} correlations found",
                    severity="warning"
                ))

        # Check metabolites data
        metab_data = self.data.get('cimametabolitestop', [])
        results.append(self._check(
            panel, "metabolites_exists", len(metab_data) > 0,
            "cimametabolitestop data missing or empty",
            fix_hint="Run: preprocess_cima_metabolites()"
        ))

        if metab_data:
            results.append(self._check(
                panel, "metabolites_count", len(metab_data) >= 100,
                f"Only {len(metab_data)} metabolite correlations (expected ≥100)",
                severity="warning"
            ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_eqtl_panel(self) -> PanelStatus:
        """Validate eQTL Browser panel."""
        panel = "eQTL"
        results = []

        # Check top eQTL data (embedded for fast load)
        top_data = self.data.get('cimaeqtltop', {})
        results.append(self._check(
            panel, "top_data_exists", bool(top_data),
            "cimaeqtltop data missing",
            fix_hint="Run: preprocess_cima_eqtl()"
        ))

        if top_data:
            eqtls = top_data.get('eqtls', [])
            results.append(self._check(
                panel, "eqtl_count", len(eqtls) >= 100,
                f"Only {len(eqtls)} eQTLs (expected ≥100)"
            ))

            # Check required fields in eQTL records
            if eqtls:
                first = eqtls[0]
                # Note: field is 'celltype' not 'cell_type' in the data
                required_fields = ['gene', 'variant', 'pvalue', 'beta', 'celltype']
                for field in required_fields:
                    results.append(self._check(
                        panel, f"eqtl_has_{field}", field in first,
                        f"eQTL records missing '{field}' field"
                    ))

        # Check full eQTL data file exists
        full_eqtl_path = VIZ_DATA_DIR / 'cima_eqtl.json'
        results.append(self._check(
            panel, "full_data_file", full_eqtl_path.exists(),
            "Full eQTL data file (cima_eqtl.json) not found",
            severity="warning",
            fix_hint="Generate full eQTL data for 'Load full data' feature"
        ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_metabolites_panel(self) -> PanelStatus:
        """Validate Metabolites Correlation panel."""
        panel = "Metabolites"
        results = []
        data = self.data.get('cimametabolitestop', [])

        results.append(self._check(
            panel, "data_exists", len(data) > 0,
            "cimametabolitestop data missing or empty",
            fix_hint="Run: preprocess_cima_metabolites()"
        ))

        if data:
            # Check count
            results.append(self._check(
                panel, "correlation_count", len(data) >= 500,
                f"Only {len(data)} correlations (expected ≥500)",
                severity="warning"
            ))

            # Check required fields
            # Note: actual field names are 'protein', 'feature' (not 'cytokine', 'metabolite')
            if data:
                first = data[0]
                required_fields = ['protein', 'feature', 'rho', 'pvalue']
                for field in required_fields:
                    results.append(self._check(
                        panel, f"has_{field}", field in first,
                        f"Metabolite records missing '{field}' field"
                    ))

                # Check for NaN values
                nan_count = sum(1 for d in data if d.get('rho') is None or str(d.get('rho')) == 'NaN')
                results.append(self._check(
                    panel, "no_nan_values", nan_count == 0,
                    f"{nan_count} records have NaN correlation values",
                    fix_hint="Filter NaN values in preprocessing"
                ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_celltype_correlations_panel(self) -> PanelStatus:
        """Validate Cell Type Correlations panel."""
        panel = "Cell Type Correlations"
        results = []
        data = self.data.get('cimacelltypecorrelations', {})

        results.append(self._check(
            panel, "data_exists", bool(data),
            "cimacelltypecorrelations data missing",
            fix_hint="Run: preprocess_cima_celltype_correlations()"
        ))

        if data:
            # Data structure: {'age': [...], 'bmi': [...]}
            # Check for age and bmi correlation arrays
            has_data = False
            for key in ['age', 'bmi', 'matrix', 'correlations']:
                if key in data:
                    val = data[key]
                    if isinstance(val, list) and len(val) > 0:
                        has_data = True
                        results.append(self._check(
                            panel, f"has_{key}_data", True,
                            f"Found {len(val)} {key} correlation records"
                        ))
                    elif isinstance(val, dict) and len(val) > 0:
                        has_data = True

            if isinstance(data, list) and len(data) > 0:
                has_data = True

            results.append(self._check(
                panel, "has_correlation_data", has_data,
                "No correlation data found (expected 'age', 'bmi', or 'matrix' key)"
            ))

            # Check record structure if we have list data
            for key in ['age', 'bmi']:
                if key in data and isinstance(data[key], list) and data[key]:
                    first = data[key][0]
                    required = ['cell_type', 'protein', 'rho', 'pvalue']
                    for field in required:
                        results.append(self._check(
                            panel, f"{key}_has_{field}", field in first,
                            f"{key} records missing '{field}' field",
                            severity="warning"
                        ))
                    break  # Only check once

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_age_bmi_panel(self) -> PanelStatus:
        """Validate Age & BMI Correlations panel."""
        panel = "Age & BMI"
        results = []
        data = self.data.get('cimacorrelations', {})

        results.append(self._check(
            panel, "data_exists", bool(data),
            "cimacorrelations data missing",
            fix_hint="Run: preprocess_cima_correlations()"
        ))

        if data:
            # Check age correlations
            age_data = data.get('age', [])
            results.append(self._check(
                panel, "has_age_correlations", len(age_data) > 0,
                f"No age correlations found"
            ))
            if age_data:
                results.append(self._check(
                    panel, "age_correlation_count", len(age_data) >= 20,
                    f"Only {len(age_data)} age correlations (expected ≥20)",
                    severity="warning"
                ))

            # Check BMI correlations
            bmi_data = data.get('bmi', [])
            results.append(self._check(
                panel, "has_bmi_correlations", len(bmi_data) > 0,
                f"No BMI correlations found"
            ))
            if bmi_data:
                results.append(self._check(
                    panel, "bmi_correlation_count", len(bmi_data) >= 20,
                    f"Only {len(bmi_data)} BMI correlations (expected ≥20)",
                    severity="warning"
                ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_age_bmi_boxplots_panel(self) -> PanelStatus:
        """Validate Age/BMI Stratified Boxplots panel."""
        panel = "Age/BMI Boxplots"
        results = []
        data = self.data.get('agebmiboxplots', {})

        results.append(self._check(
            panel, "data_exists", bool(data),
            "agebmiboxplots data missing",
            fix_hint="Run: preprocess_age_bmi_boxplots()"
        ))

        if data:
            # Data structure: {'cima': {'age': [...], 'bmi': [...], 'cytosig_signatures': [...]}}
            cima_data = data.get('cima', {})
            results.append(self._check(
                panel, "has_cima_data", bool(cima_data),
                "No CIMA data found in boxplots"
            ))

            if cima_data:
                # Check for age data
                age_data = cima_data.get('age', [])
                results.append(self._check(
                    panel, "has_age_data", len(age_data) > 0,
                    "No age stratified data found"
                ))

                # Check for BMI data
                bmi_data = cima_data.get('bmi', [])
                results.append(self._check(
                    panel, "has_bmi_data", len(bmi_data) > 0,
                    "No BMI stratified data found"
                ))

                # Check signatures lists
                cytosig_sigs = cima_data.get('cytosig_signatures', [])
                results.append(self._check(
                    panel, "has_cytosig_signatures", len(cytosig_sigs) > 0,
                    "No CytoSig signatures list found",
                    severity="warning"
                ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_biochemistry_panel(self) -> PanelStatus:
        """Validate Biochemistry Correlations panel."""
        panel = "Biochemistry"
        results = []
        data = self.data.get('cimacorrelations', {})

        results.append(self._check(
            panel, "data_exists", bool(data),
            "cimacorrelations data missing",
            fix_hint="Run: preprocess_cima_correlations()"
        ))

        if data:
            # Check biochemistry correlations
            biochem_data = data.get('biochemistry', [])
            results.append(self._check(
                panel, "has_biochemistry", len(biochem_data) > 0,
                "No biochemistry correlations found"
            ))

            if biochem_data:
                results.append(self._check(
                    panel, "biochemistry_count", len(biochem_data) >= 50,
                    f"Only {len(biochem_data)} biochemistry correlations (expected ≥50)",
                    severity="warning"
                ))

                # Check required fields
                first = biochem_data[0]
                for field in ['protein', 'feature', 'rho', 'pvalue']:
                    results.append(self._check(
                        panel, f"biochem_has_{field}", field in first,
                        f"Biochemistry records missing '{field}' field"
                    ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    def validate_differential_panel(self) -> PanelStatus:
        """Validate Differential Analysis panel."""
        panel = "Differential"
        results = []
        data = self.data.get('cimadifferential')

        results.append(self._check(
            panel, "data_exists", bool(data),
            "cimadifferential data missing",
            fix_hint="Run: preprocess_cima_differential()"
        ))

        if data:
            # Data is a list of differential records [{protein, comparison, activity_diff, pvalue, ...}, ...]
            if isinstance(data, list):
                results.append(self._check(
                    panel, "record_count", len(data) >= 100,
                    f"Only {len(data)} differential records (expected ≥100)"
                ))

                if data:
                    first = data[0]
                    # Check required fields
                    required_fields = ['protein', 'comparison', 'activity_diff', 'pvalue']
                    for field in required_fields:
                        results.append(self._check(
                            panel, f"has_{field}", field in first,
                            f"Differential records missing '{field}' field"
                        ))

                    # Check for sex comparison
                    comparisons = set(d.get('comparison', '') for d in data)
                    has_sex = any('sex' in c.lower() or 'male' in c.lower() for c in comparisons)
                    results.append(self._check(
                        panel, "has_sex_comparison", has_sex,
                        f"No sex comparison found. Comparisons: {list(comparisons)[:3]}",
                        severity="warning"
                    ))

            elif isinstance(data, dict):
                # Alternative dict format with 'sex', 'smoking' keys
                sex_data = data.get('sex', [])
                results.append(self._check(
                    panel, "has_sex_differential", len(sex_data) > 0,
                    "No sex differential data found"
                ))

        functional = all(r.passed for r in results if r.severity == "error")
        return PanelStatus(panel, functional, results)

    # =========================================================================
    # Main Validation
    # =========================================================================

    def validate_all(self) -> Dict[str, PanelStatus]:
        """Run all panel validations."""
        print("\n" + "="*60)
        print("CIMA Panel Validation Report")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")

        validators = [
            # 10 CIMA panels
            self.validate_age_bmi_panel,
            self.validate_age_bmi_boxplots_panel,
            self.validate_biochemistry_panel,
            self.validate_biochem_scatter_panel,
            self.validate_metabolites_panel,
            self.validate_differential_panel,
            self.validate_celltype_panel,
            self.validate_multiomics_panel,
            self.validate_population_panel,
            self.validate_eqtl_panel,
        ]

        for validator in validators:
            status = validator()
            self.panel_statuses[status.name] = status

        return self.panel_statuses

    def validate_panel(self, panel_name: str) -> Optional[PanelStatus]:
        """Validate a specific panel."""
        panel_map = {
            # 10 CIMA panels with aliases
            'age_bmi': self.validate_age_bmi_panel,
            'age-bmi': self.validate_age_bmi_panel,
            'age_bmi_boxplots': self.validate_age_bmi_boxplots_panel,
            'age-bmi-boxplots': self.validate_age_bmi_boxplots_panel,
            'boxplots': self.validate_age_bmi_boxplots_panel,
            'biochemistry': self.validate_biochemistry_panel,
            'biochem': self.validate_biochem_scatter_panel,
            'biochem_scatter': self.validate_biochem_scatter_panel,
            'metabolites': self.validate_metabolites_panel,
            'differential': self.validate_differential_panel,
            'celltype': self.validate_celltype_panel,
            'cell_type': self.validate_celltype_panel,
            'cell_types': self.validate_celltype_panel,
            'multiomics': self.validate_multiomics_panel,
            'multi_omics': self.validate_multiomics_panel,
            'population': self.validate_population_panel,
            'eqtl': self.validate_eqtl_panel,
        }

        validator = panel_map.get(panel_name.lower().replace('-', '_'))
        if validator:
            return validator()
        else:
            print(f"Unknown panel: {panel_name}")
            print(f"Available panels: {list(panel_map.keys())}")
            return None

    def generate_report(self) -> str:
        """Generate a summary report."""
        lines = [
            "\n" + "="*60,
            "VALIDATION SUMMARY",
            "="*60,
        ]

        total_errors = 0
        total_warnings = 0

        for name, status in self.panel_statuses.items():
            icon = "✓" if status.functional else "✗"
            errors = len(status.errors)
            warnings = len(status.warnings)
            total_errors += errors
            total_warnings += warnings

            lines.append(f"\n{icon} {name}: {'PASS' if status.functional else 'FAIL'}")
            if errors:
                lines.append(f"   Errors: {errors}")
            if warnings:
                lines.append(f"   Warnings: {warnings}")

        lines.append("\n" + "-"*60)
        lines.append(f"Total: {len(self.panel_statuses)} panels")
        lines.append(f"Functional: {sum(1 for s in self.panel_statuses.values() if s.functional)}")
        lines.append(f"Errors: {total_errors}, Warnings: {total_warnings}")
        lines.append("="*60 + "\n")

        return "\n".join(lines)

    def get_fix_recommendations(self) -> List[Dict[str, Any]]:
        """Get list of recommended fixes."""
        fixes = []
        for name, status in self.panel_statuses.items():
            for result in status.results:
                if not result.passed and result.fix_hint:
                    fixes.append({
                        'panel': name,
                        'issue': result.check_name,
                        'message': result.message,
                        'severity': result.severity,
                        'fix_hint': result.fix_hint
                    })
        return fixes

    def save_report(self, filepath: Path = None):
        """Save validation report to file."""
        if filepath is None:
            filepath = LOG_DIR / f"cima_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'panels': {},
            'fixes_needed': self.get_fix_recommendations()
        }

        for name, status in self.panel_statuses.items():
            report['panels'][name] = {
                'functional': status.functional,
                'errors': [{'check': r.check_name, 'message': r.message} for r in status.errors],
                'warnings': [{'check': r.check_name, 'message': r.message} for r in status.warnings],
            }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description='CIMA Panel Validator')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--panel', '-p', type=str, help='Validate specific panel')
    parser.add_argument('--fix', action='store_true', help='Show fix recommendations')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    args = parser.parse_args()

    validator = CIMAPanelValidator(verbose=args.verbose)

    if not validator.load_data():
        print("Failed to load data. Exiting.")
        sys.exit(1)

    if args.panel:
        status = validator.validate_panel(args.panel)
        if status:
            validator.panel_statuses[status.name] = status
    else:
        validator.validate_all()

    print(validator.generate_report())

    if args.fix:
        fixes = validator.get_fix_recommendations()
        if fixes:
            print("\nRECOMMENDED FIXES:")
            print("-"*40)
            for fix in fixes:
                print(f"\n[{fix['severity'].upper()}] {fix['panel']} - {fix['issue']}")
                print(f"  Issue: {fix['message']}")
                print(f"  Fix: {fix['fix_hint']}")

    if args.save:
        validator.save_report()

    # Exit with error code if any panel is non-functional
    if any(not s.functional for s in validator.panel_statuses.values()):
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()


def check_json_files_for_nan():
    """Check all JSON files for NaN values (invalid in JSON)."""
    import re
    from pathlib import Path
    
    DATA_DIR = Path('/vf/users/parks34/projects/2cytoatlas/visualization/data')
    issues = []
    
    for json_file in DATA_DIR.glob('*.json'):
        content = json_file.read_text()
        nan_count = len(re.findall(r'\bNaN\b', content))
        inf_count = len(re.findall(r'\b-?Infinity\b', content))
        
        if nan_count > 0 or inf_count > 0:
            issues.append({
                'file': json_file.name,
                'nan_count': nan_count,
                'inf_count': inf_count
            })
    
    return issues


def fix_json_nan_values():
    """Fix NaN and Infinity values in all JSON files."""
    import re
    from pathlib import Path
    
    DATA_DIR = Path('/vf/users/parks34/projects/2cytoatlas/visualization/data')
    fixed = []
    
    for json_file in DATA_DIR.glob('*.json'):
        content = json_file.read_text()
        original = content
        
        # Replace NaN and Infinity with null
        content = re.sub(r'\bNaN\b', 'null', content)
        content = re.sub(r'\b-?Infinity\b', 'null', content)
        
        if content != original:
            json_file.write_text(content)
            fixed.append(json_file.name)
    
    return fixed


if __name__ == '__main__' and '--check-nan' in sys.argv:
    issues = check_json_files_for_nan()
    if issues:
        print("Found NaN/Infinity values in JSON files:")
        for issue in issues:
            print(f"  {issue['file']}: {issue['nan_count']} NaN, {issue['inf_count']} Infinity")
        print("\nRun with --fix-nan to fix these issues")
    else:
        print("No NaN/Infinity values found in JSON files")

if __name__ == '__main__' and '--fix-nan' in sys.argv:
    fixed = fix_json_nan_values()
    if fixed:
        print(f"Fixed NaN/Infinity values in: {fixed}")
    else:
        print("No files needed fixing")
