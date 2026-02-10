# maintenance

Run codebase audit and report findings

## Instructions

When the user invokes /maintenance, perform the following:

1. **Run the audit script**:
   ```bash
   cd /data/parks34/projects/2cytoatlas
   python scripts/maintenance/audit_clutter.py --report
   ```

2. **Run quick checks**:
   ```bash
   # __pycache__ count
   find . -name "__pycache__" -type d | wc -l
   # Tracked file counts by area
   for d in scripts archive agents .claude docs; do echo "$d: $(git ls-files $d/ | wc -l)"; done
   # Settings line count
   wc -l .claude/settings.local.json
   ```

3. **Report findings** with:
   - Summary of each audit category
   - Actionable recommendations
   - Suggested cleanup commands

4. **If `--fix` argument provided**, run safe cleanup:
   ```bash
   find . -name "__pycache__" -type d -exec rm -rf {} +
   find . -name ".DS_Store" -delete
   ```

## Arguments

- `/maintenance` - Run audit and report findings
- `/maintenance --fix` - Run audit and apply safe fixes
- `/maintenance --json` - Output JSON report

## Example Usage

```
/maintenance
/maintenance --fix
```
