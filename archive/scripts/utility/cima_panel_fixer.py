#!/usr/bin/env python3
"""
CIMA Panel Fixer Agent

Automatically fixes common issues in CIMA visualization panels.
Works in conjunction with cima_panel_validator.py.

Usage:
    python cima_panel_fixer.py [--panel PANEL_NAME] [--dry-run] [--all]
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Paths
SCRIPTS_DIR = Path('/vf/users/parks34/projects/2secactpy/scripts')
VIZ_DATA_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')
RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results')
LOG_DIR = Path('/vf/users/parks34/projects/2secactpy/logs')
LOG_DIR.mkdir(exist_ok=True)


class CIMAPanelFixer:
    """Automatically fixes CIMA panel issues."""

    def __init__(self, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.fixes_applied = []
        self.fixes_failed = []

    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(msg)

    def run_preprocessing(self, func_name: str) -> bool:
        """Run a preprocessing function from 06_preprocess_viz_data.py."""
        self.log(f"\n{'[DRY RUN] Would run' if self.dry_run else 'Running'}: {func_name}()")

        if self.dry_run:
            return True

        cmd = f'''
source ~/bin/myconda && conda activate secactpy && cd /data/parks34/projects/2secactpy && python3 << 'EOF'
import sys
sys.path.insert(0, '/vf/users/parks34/projects/2secactpy/scripts')
exec(open('/vf/users/parks34/projects/2secactpy/scripts/06_preprocess_viz_data.py').read())
{func_name}()
EOF
'''
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                self.log(f"  ✓ {func_name}() completed successfully")
                return True
            else:
                self.log(f"  ✗ {func_name}() failed: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            self.log(f"  ✗ {func_name}() timed out")
            return False
        except Exception as e:
            self.log(f"  ✗ {func_name}() error: {e}")
            return False

    def regenerate_embedded_data(self) -> bool:
        """Regenerate embedded_data.js from all JSON files."""
        self.log(f"\n{'[DRY RUN] Would regenerate' if self.dry_run else 'Regenerating'} embedded_data.js")

        if self.dry_run:
            return True

        script = '''
import json
from pathlib import Path

DATA_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

json_files = list(DATA_DIR.glob('*.json'))
print(f"Found {len(json_files)} JSON files")

all_data = {}
key_mapping = {
    'cima_biochem_scatter.json': 'cimabiochemscatter',
    'cima_celltype.json': 'cimacelltype',
    'cima_correlations.json': 'cimacorrelations',
    'cima_metabolites_top.json': 'cimametabolitestop',
    'age_bmi_boxplots.json': 'agebmiboxplots',
    'cross_atlas.json': 'crossatlas',
    'cima_population_stratification.json': 'cimapopulationstratification',
    'cima_eqtl.json': 'cimaeqtl',
    'cima_eqtl_top.json': 'cimaeqtltop',
    'cima_celltype_correlations.json': 'cimacelltypecorrelations',
    'immune_infiltration.json': 'immuneinfiltration',
    'exhaustion.json': 'exhaustion',
    'caf_signatures.json': 'cafsignatures',
    'adjacent_tissue.json': 'adjacenttissue',
    'cancer_types.json': 'cancertypes',
}

for jf in json_files:
    key = key_mapping.get(jf.name, jf.stem.replace('_', '').replace('-', ''))
    try:
        with open(jf) as f:
            all_data[key] = json.load(f)
        print(f"  Loaded {jf.name} as '{key}'")
    except Exception as e:
        print(f"  Error loading {jf.name}: {e}")

output_file = DATA_DIR / 'embedded_data.js'
with open(output_file, 'w') as f:
    f.write('const EMBEDDED_DATA = ')
    json.dump(all_data, f, separators=(',', ':'))
    f.write(';')

print(f"\\nWrote {output_file.name}: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
'''

        cmd = f"source ~/bin/myconda && conda activate secactpy && python3 -c '''{script}'''"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.log(f"  ✓ embedded_data.js regenerated")
                self.log(result.stdout)
                return True
            else:
                self.log(f"  ✗ Failed to regenerate embedded_data.js: {result.stderr[:200]}")
                return False
        except Exception as e:
            self.log(f"  ✗ Error: {e}")
            return False

    # =========================================================================
    # Panel-specific fixers (10 CIMA panels)
    # =========================================================================

    def fix_age_bmi_panel(self) -> bool:
        """Fix Age & BMI Correlations panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Age & BMI Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_correlations')
        if success:
            self.fixes_applied.append(('Age & BMI', 'preprocess_cima_correlations'))
        else:
            self.fixes_failed.append(('Age & BMI', 'preprocess_cima_correlations'))
        return success

    def fix_age_bmi_boxplots_panel(self) -> bool:
        """Fix Age/BMI Stratified Boxplots panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Age/BMI Boxplots Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_age_bmi_boxplots')
        if success:
            self.fixes_applied.append(('Age/BMI Boxplots', 'preprocess_age_bmi_boxplots'))
        else:
            self.fixes_failed.append(('Age/BMI Boxplots', 'preprocess_age_bmi_boxplots'))
        return success

    def fix_biochemistry_panel(self) -> bool:
        """Fix Biochemistry Correlations panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Biochemistry Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_correlations')
        if success:
            self.fixes_applied.append(('Biochemistry', 'preprocess_cima_correlations'))
        else:
            self.fixes_failed.append(('Biochemistry', 'preprocess_cima_correlations'))
        return success

    def fix_differential_panel(self) -> bool:
        """Fix Differential Analysis panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Differential Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_differential')
        if success:
            self.fixes_applied.append(('Differential', 'preprocess_cima_differential'))
        else:
            self.fixes_failed.append(('Differential', 'preprocess_cima_differential'))
        return success

    def fix_celltype_panel(self) -> bool:
        """Fix Cell Type Overview panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Cell Type Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_celltype')
        if success:
            self.fixes_applied.append(('Cell Type', 'preprocess_cima_celltype'))
        else:
            self.fixes_failed.append(('Cell Type', 'preprocess_cima_celltype'))
        return success

    def fix_biochem_scatter_panel(self) -> bool:
        """Fix Biochemistry Scatter panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Biochem Scatter Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_biochem_scatter')
        if success:
            self.fixes_applied.append(('Biochem Scatter', 'preprocess_cima_biochem_scatter'))
        else:
            self.fixes_failed.append(('Biochem Scatter', 'preprocess_cima_biochem_scatter'))
        return success

    def fix_population_panel(self) -> bool:
        """Fix Population Stratification panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Population Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_population_stratification')
        if success:
            self.fixes_applied.append(('Population', 'preprocess_cima_population_stratification'))
        else:
            self.fixes_failed.append(('Population', 'preprocess_cima_population_stratification'))
        return success

    def fix_multiomics_panel(self) -> bool:
        """Fix Multi-omics Integration panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Multi-omics Panel")
        self.log("="*50)

        success1 = self.run_preprocessing('preprocess_cima_correlations')
        success2 = self.run_preprocessing('preprocess_cima_metabolites')

        if success1:
            self.fixes_applied.append(('Multi-omics', 'preprocess_cima_correlations'))
        else:
            self.fixes_failed.append(('Multi-omics', 'preprocess_cima_correlations'))

        if success2:
            self.fixes_applied.append(('Multi-omics', 'preprocess_cima_metabolites'))
        else:
            self.fixes_failed.append(('Multi-omics', 'preprocess_cima_metabolites'))

        return success1 and success2

    def fix_eqtl_panel(self) -> bool:
        """Fix eQTL Browser panel."""
        self.log("\n" + "="*50)
        self.log("Fixing eQTL Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_eqtl')
        if success:
            self.fixes_applied.append(('eQTL', 'preprocess_cima_eqtl'))
        else:
            self.fixes_failed.append(('eQTL', 'preprocess_cima_eqtl'))
        return success

    def fix_metabolites_panel(self) -> bool:
        """Fix Metabolites Correlation panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Metabolites Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_metabolites')
        if success:
            self.fixes_applied.append(('Metabolites', 'preprocess_cima_metabolites'))
        else:
            self.fixes_failed.append(('Metabolites', 'preprocess_cima_metabolites'))
        return success

    def fix_celltype_correlations_panel(self) -> bool:
        """Fix Cell Type Correlations panel."""
        self.log("\n" + "="*50)
        self.log("Fixing Cell Type Correlations Panel")
        self.log("="*50)

        success = self.run_preprocessing('preprocess_cima_celltype_correlations')
        if success:
            self.fixes_applied.append(('Cell Type Correlations', 'preprocess_cima_celltype_correlations'))
        else:
            self.fixes_failed.append(('Cell Type Correlations', 'preprocess_cima_celltype_correlations'))
        return success

    # =========================================================================
    # Main fix methods
    # =========================================================================

    def fix_panel(self, panel_name: str) -> bool:
        """Fix a specific panel."""
        panel_map = {
            # 10 CIMA panels with aliases
            'age_bmi': self.fix_age_bmi_panel,
            'age-bmi': self.fix_age_bmi_panel,
            'age_bmi_boxplots': self.fix_age_bmi_boxplots_panel,
            'age-bmi-boxplots': self.fix_age_bmi_boxplots_panel,
            'boxplots': self.fix_age_bmi_boxplots_panel,
            'biochemistry': self.fix_biochemistry_panel,
            'biochem': self.fix_biochem_scatter_panel,
            'biochem_scatter': self.fix_biochem_scatter_panel,
            'metabolites': self.fix_metabolites_panel,
            'differential': self.fix_differential_panel,
            'celltype': self.fix_celltype_panel,
            'cell_type': self.fix_celltype_panel,
            'cell_types': self.fix_celltype_panel,
            'multiomics': self.fix_multiomics_panel,
            'multi_omics': self.fix_multiomics_panel,
            'population': self.fix_population_panel,
            'eqtl': self.fix_eqtl_panel,
        }

        fixer = panel_map.get(panel_name.lower().replace('-', '_'))
        if fixer:
            return fixer()
        else:
            print(f"Unknown panel: {panel_name}")
            print(f"Available panels: {list(panel_map.keys())}")
            return False

    def fix_all(self) -> bool:
        """Fix all 10 CIMA panels."""
        self.log("\n" + "="*60)
        self.log("CIMA Panel Fixer - Fixing All 10 Panels")
        self.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("="*60)

        all_success = True
        fixers = [
            # 10 CIMA panels in tab order
            self.fix_age_bmi_panel,
            self.fix_age_bmi_boxplots_panel,
            self.fix_biochemistry_panel,
            self.fix_biochem_scatter_panel,
            self.fix_metabolites_panel,
            self.fix_differential_panel,
            self.fix_celltype_panel,
            self.fix_multiomics_panel,
            self.fix_population_panel,
            self.fix_eqtl_panel,
        ]

        for fixer in fixers:
            try:
                success = fixer()
                all_success = all_success and success
            except Exception as e:
                self.log(f"Error in {fixer.__name__}: {e}")
                all_success = False

        # Regenerate embedded data after all fixes
        if not self.dry_run:
            self.log("\n" + "="*50)
            self.log("Finalizing: Regenerating embedded_data.js")
            self.log("="*50)
            self.regenerate_embedded_data()

        return all_success

    def fix_from_validation(self, validation_report: Dict[str, Any]) -> bool:
        """Fix panels based on validation report."""
        fixes_needed = validation_report.get('fixes_needed', [])

        if not fixes_needed:
            self.log("No fixes needed based on validation report.")
            return True

        self.log(f"\nFound {len(fixes_needed)} issues to fix")

        # Group fixes by panel
        panels_to_fix = set()
        for fix in fixes_needed:
            if fix['severity'] == 'error':
                panels_to_fix.add(fix['panel'].lower().replace(' ', '_'))

        self.log(f"Panels to fix: {panels_to_fix}")

        all_success = True
        for panel in panels_to_fix:
            success = self.fix_panel(panel)
            all_success = all_success and success

        # Regenerate embedded data
        if not self.dry_run and panels_to_fix:
            self.regenerate_embedded_data()

        return all_success

    def generate_report(self) -> str:
        """Generate fix report."""
        lines = [
            "\n" + "="*60,
            "FIX SUMMARY",
            "="*60,
            f"\nFixes Applied: {len(self.fixes_applied)}",
        ]

        for panel, func in self.fixes_applied:
            lines.append(f"  ✓ {panel}: {func}")

        if self.fixes_failed:
            lines.append(f"\nFixes Failed: {len(self.fixes_failed)}")
            for panel, func in self.fixes_failed:
                lines.append(f"  ✗ {panel}: {func}")

        lines.append("\n" + "="*60)
        return "\n".join(lines)


def run_validation_and_fix():
    """Run validation, then fix any issues found."""
    print("="*60)
    print("CIMA Panel Auto-Fix Pipeline")
    print("="*60)

    # Step 1: Run validation
    print("\n[Step 1] Running validation...")
    from cima_panel_validator import CIMAPanelValidator

    validator = CIMAPanelValidator(verbose=True)
    if not validator.load_data():
        print("Failed to load data for validation")
        return False

    validator.validate_all()
    print(validator.generate_report())

    # Check if any panels need fixing
    panels_needing_fix = [name for name, status in validator.panel_statuses.items()
                         if not status.functional]

    if not panels_needing_fix:
        print("\n✓ All panels are functional. No fixes needed.")
        return True

    # Step 2: Fix broken panels
    print(f"\n[Step 2] Fixing {len(panels_needing_fix)} panel(s): {panels_needing_fix}")

    fixer = CIMAPanelFixer(dry_run=False, verbose=True)
    for panel in panels_needing_fix:
        fixer.fix_panel(panel.lower().replace(' ', '_'))

    # Regenerate embedded data
    fixer.regenerate_embedded_data()

    print(fixer.generate_report())

    # Step 3: Re-validate
    print("\n[Step 3] Re-validating after fixes...")
    validator2 = CIMAPanelValidator(verbose=False)
    validator2.load_data()
    validator2.validate_all()

    still_broken = [name for name, status in validator2.panel_statuses.items()
                    if not status.functional]

    if still_broken:
        print(f"\n⚠ {len(still_broken)} panel(s) still need attention: {still_broken}")
        return False
    else:
        print("\n✓ All panels are now functional!")
        return True


def main():
    parser = argparse.ArgumentParser(description='CIMA Panel Fixer')
    parser.add_argument('--panel', '-p', type=str, help='Fix specific panel')
    parser.add_argument('--all', '-a', action='store_true', help='Fix all panels')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Show what would be done')
    parser.add_argument('--auto', action='store_true',
                        help='Run validation and automatically fix issues')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    args = parser.parse_args()

    if args.auto:
        success = run_validation_and_fix()
        sys.exit(0 if success else 1)

    fixer = CIMAPanelFixer(dry_run=args.dry_run, verbose=not args.quiet)

    if args.panel:
        success = fixer.fix_panel(args.panel)
        if not args.dry_run:
            fixer.regenerate_embedded_data()
    elif args.all:
        success = fixer.fix_all()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python cima_panel_fixer.py --auto          # Validate and auto-fix")
        print("  python cima_panel_fixer.py --panel population")
        print("  python cima_panel_fixer.py --all --dry-run")
        sys.exit(0)

    print(fixer.generate_report())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
