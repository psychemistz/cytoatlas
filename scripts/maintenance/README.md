# Maintenance Scripts

Tooling for routine codebase audits and equivalence testing.

## Scripts

### `audit_clutter.py`

Programmatic audit that reports codebase clutter:

```bash
# Human-readable summary
python scripts/maintenance/audit_clutter.py --report

# JSON report
python scripts/maintenance/audit_clutter.py --json
```

**Reports:**
- `__pycache__` directories and `.pyc` files
- Tracked files in `archive/` with line counts
- Empty directories (only `.gitkeep`)
- Analysis scripts NOT importing from `cytoatlas_pipeline`
- Stale logs (>30 days)
- Permission bloat in `.claude/settings.local.json`

### `equivalence_test.py`

Test harness to compare script output vs pipeline output:

```bash
# Compare CSV/TSV DataFrames
python scripts/maintenance/equivalence_test.py compare-df script.csv pipeline.csv --tolerance 1e-6

# Compare JSON files
python scripts/maintenance/equivalence_test.py compare-json script.json pipeline.json

# Compare H5AD files
python scripts/maintenance/equivalence_test.py compare-h5ad script.h5ad pipeline.h5ad
```

**As a library:**
```python
from scripts.maintenance.equivalence_test import compare_dataframes, compare_json_files, compare_h5ad_files

result = compare_dataframes(df_script, df_pipeline, tolerance=1e-6)
print(result)       # Human-readable summary
assert result.equal  # Use in tests
```

## Routine Maintenance Checklist

Run after major updates or every ~50 commits:

### Quick Check (5 min)
```bash
find . -name "__pycache__" -type d | wc -l
grep -rL "cytoatlas_pipeline" scripts/[0-9]*.py 2>/dev/null | wc -l
for d in scripts archive agents .claude docs; do echo "$d: $(git ls-files $d/ | wc -l)"; done
```

### Deep Check (30 min)
```bash
python scripts/maintenance/audit_clutter.py --report
pytest cytoatlas-pipeline/tests/equivalence/ -v --tb=short
```

### Cleanup Actions
```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name ".DS_Store" -delete
find logs/ -mtime +30 \( -name "*.out" -o -name "*.err" \) -delete
```
