#!/usr/bin/env python3
"""
Audit script for routine codebase clutter detection.

Reports:
- __pycache__ directories and .pyc files
- Tracked files in archive/ with line counts
- Empty directories (only .gitkeep)
- Analysis scripts NOT importing from cytoatlas_pipeline
- Stale logs (>30 days)
- Permission bloat in .claude/settings.local.json

Usage:
    python scripts/maintenance/audit_clutter.py [--report] [--json]

Options:
    --report    Print human-readable summary (default)
    --json      Output JSON report to stdout
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def get_project_root() -> Path:
    """Find project root by looking for .git directory."""
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find project root (.git directory)")


def find_pycache_dirs(root: Path) -> list[dict]:
    """Find __pycache__ directories and count .pyc files."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "__pycache__" in os.path.basename(dirpath):
            pyc_count = sum(1 for f in filenames if f.endswith((".pyc", ".pyo")))
            results.append({
                "path": os.path.relpath(dirpath, root),
                "pyc_files": pyc_count,
            })
    return results


def get_tracked_archive_files(root: Path) -> list[dict]:
    """List tracked files in archive/ with line counts."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "archive/"],
            cwd=root, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return []

    results = []
    for filepath in output.splitlines():
        if not filepath:
            continue
        full_path = root / filepath
        try:
            line_count = sum(1 for _ in open(full_path, "r", errors="replace"))
        except (OSError, UnicodeDecodeError):
            line_count = -1
        results.append({"path": filepath, "lines": line_count})
    return results


def find_gitkeep_only_dirs(root: Path) -> list[str]:
    """Find directories that contain only a .gitkeep file."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirpath.split(os.sep):
            continue
        if filenames == [".gitkeep"] and not dirnames:
            results.append(os.path.relpath(dirpath, root))
    return results


def find_scripts_without_pipeline_import(root: Path) -> list[str]:
    """Find numbered analysis scripts that don't import cytoatlas_pipeline."""
    scripts_dir = root / "scripts"
    results = []
    pattern = re.compile(r"^\d+_.*\.py$")

    for f in sorted(scripts_dir.iterdir()):
        if not f.is_file() or not pattern.match(f.name):
            continue
        try:
            content = f.read_text(errors="replace")
            if "cytoatlas_pipeline" not in content:
                results.append(f"scripts/{f.name}")
        except OSError:
            pass
    return results


def find_stale_logs(root: Path, max_age_days: int = 30) -> list[dict]:
    """Find log files older than max_age_days."""
    logs_dir = root / "logs"
    if not logs_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=max_age_days)
    results = []
    for f in logs_dir.iterdir():
        if f.name == ".gitkeep":
            continue
        if f.is_file():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                results.append({
                    "path": f"logs/{f.name}",
                    "modified": mtime.isoformat(),
                    "age_days": (datetime.now() - mtime).days,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
    return results


def audit_settings_bloat(root: Path) -> dict:
    """Check .claude/settings.local.json for permission bloat."""
    settings_path = root / ".claude" / "settings.local.json"
    if not settings_path.exists():
        return {"exists": False, "total_entries": 0, "categories": {}}

    try:
        data = json.loads(settings_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"exists": True, "parse_error": True}

    allow_list = data.get("permissions", {}).get("allow", [])
    deny_list = data.get("permissions", {}).get("deny", [])

    # Categorize allow entries
    categories: dict[str, list[str]] = {
        "shell_fragments": [],
        "one_off_commands": [],
        "git_specific": [],
        "heredoc_patterns": [],
        "standard_tools": [],
    }

    shell_keywords = {"do", "then", "fi", "done", "else", "for", "if", "while"}
    for entry in allow_list:
        # Extract the command part
        m = re.match(r"Bash\((.+)\)", entry)
        if not m:
            categories["standard_tools"].append(entry)
            continue

        cmd = m.group(1).rstrip(":*").strip()

        if cmd in shell_keywords or cmd.startswith("do "):
            categories["shell_fragments"].append(entry)
        elif "git -C" in entry:
            categories["git_specific"].append(entry)
        elif "<<" in entry or "EOF" in entry or "/tmp/" in entry:
            categories["heredoc_patterns"].append(entry)
        elif len(entry) > 120 or "=" in cmd:
            categories["one_off_commands"].append(entry)
        else:
            categories["standard_tools"].append(entry)

    return {
        "exists": True,
        "total_entries": len(allow_list) + len(deny_list),
        "allow_count": len(allow_list),
        "deny_count": len(deny_list),
        "categories": {k: len(v) for k, v in categories.items()},
        "removable": sum(
            len(v) for k, v in categories.items()
            if k != "standard_tools"
        ),
    }


def run_audit(root: Path) -> dict:
    """Run full audit and return results."""
    return {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(root),
        "pycache_dirs": find_pycache_dirs(root),
        "archive_files": get_tracked_archive_files(root),
        "gitkeep_only_dirs": find_gitkeep_only_dirs(root),
        "scripts_without_pipeline": find_scripts_without_pipeline_import(root),
        "stale_logs": find_stale_logs(root),
        "settings_bloat": audit_settings_bloat(root),
    }


def print_summary(report: dict) -> None:
    """Print human-readable summary."""
    print("=" * 60)
    print("  CytoAtlas Codebase Audit Report")
    print(f"  {report['timestamp']}")
    print("=" * 60)

    # __pycache__
    pycache = report["pycache_dirs"]
    total_pyc = sum(d["pyc_files"] for d in pycache)
    print(f"\n1. __pycache__ directories: {len(pycache)} ({total_pyc} .pyc files)")
    for d in pycache[:5]:
        print(f"   - {d['path']} ({d['pyc_files']} files)")
    if len(pycache) > 5:
        print(f"   ... and {len(pycache) - 5} more")

    # Archive files
    archive = report["archive_files"]
    total_lines = sum(f["lines"] for f in archive if f["lines"] > 0)
    print(f"\n2. Tracked archive files: {len(archive)} ({total_lines:,} lines)")

    # gitkeep-only dirs
    gitkeep = report["gitkeep_only_dirs"]
    print(f"\n3. Empty directories (gitkeep only): {len(gitkeep)}")
    for d in gitkeep:
        print(f"   - {d}")

    # Scripts without pipeline
    scripts = report["scripts_without_pipeline"]
    print(f"\n4. Analysis scripts NOT using cytoatlas_pipeline: {len(scripts)}")
    for s in scripts:
        print(f"   - {s}")

    # Stale logs
    stale = report["stale_logs"]
    total_size = sum(l["size_kb"] for l in stale)
    print(f"\n5. Stale logs (>30 days): {len(stale)} ({total_size:.0f} KB)")
    for l in stale[:5]:
        print(f"   - {l['path']} ({l['age_days']}d old, {l['size_kb']} KB)")
    if len(stale) > 5:
        print(f"   ... and {len(stale) - 5} more")

    # Settings bloat
    settings = report["settings_bloat"]
    if settings.get("exists"):
        print(f"\n6. Settings permissions: {settings.get('allow_count', 0)} allow, {settings.get('deny_count', 0)} deny")
        cats = settings.get("categories", {})
        for cat, count in cats.items():
            print(f"   - {cat}: {count}")
        print(f"   Removable entries: ~{settings.get('removable', 0)}")
    else:
        print("\n6. Settings file: not found")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Audit codebase clutter")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Print human-readable summary (default)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON report")
    args = parser.parse_args()

    root = get_project_root()
    report = run_audit(root)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)


if __name__ == "__main__":
    main()
