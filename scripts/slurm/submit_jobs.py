#!/usr/bin/env python3
"""Generate and submit SLURM jobs from jobs.yaml configuration.

Replaces 28 individual shell scripts with a single parameterized generator.
Reads job definitions from jobs.yaml, renders them from job_template.sh,
and submits with proper dependencies.

Usage:
    # List all available jobs
    python submit_jobs.py --list

    # Dry-run: show generated scripts without submitting
    python submit_jobs.py --dry-run

    # Submit all main pipeline jobs (phases 0-3)
    python submit_jobs.py --phases 0-3

    # Submit pilot only
    python submit_jobs.py --jobs pilot

    # Submit specific jobs
    python submit_jobs.py --jobs cima,inflammation,scatlas

    # Submit all validation jobs (phases 10+)
    python submit_jobs.py --phases 10-14

    # Submit everything
    python submit_jobs.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
JOBS_YAML = SCRIPT_DIR / "jobs.yaml"
TEMPLATE = SCRIPT_DIR / "job_template.sh"


def load_config() -> dict:
    """Load jobs.yaml configuration."""
    with open(JOBS_YAML) as f:
        return yaml.safe_load(f)


def render_job_script(
    job_id: str,
    job: dict,
    profile: dict,
    common: dict,
    array_item: str | None = None,
) -> str:
    """Render a SLURM job script from the template."""
    template = TEMPLATE.read_text()

    # Resolve array variables
    args = job.get("args", "")
    log_prefix = job_id
    if array_item:
        # Parse compound array items like "main_L1" → cohort=main, level=L1
        array_var = job.get("array_var", "")
        if array_var == "cohort_level":
            parts = array_item.rsplit("_", 1)
            cohort, level = parts[0], parts[1]
            args = args.replace("{cohort}", cohort).replace("{level}", level)
            log_prefix = f"{job_id}_{cohort}_{level}"
        elif array_var == "dataset_level":
            # e.g., "normal_organ_celltype" → dataset=normal, level=organ_celltype
            parts = array_item.split("_", 1)
            dataset, level = parts[0], parts[1]
            args = args.replace("{dataset}", dataset).replace("{level}", level)
            log_prefix = f"{job_id}_{dataset}_{level}"
        elif array_var == "level":
            args = args.replace("{level}", array_item)
            log_prefix = f"{job_id}_{array_item}"

    # Module lines
    modules = profile.get("modules", [])
    if modules:
        module_lines = "\n".join(f"module load {m}" for m in modules)
    else:
        module_lines = "# No modules required"

    # GRES line
    gres = profile.get("gres", "")
    gres_line = f"#SBATCH --gres={gres}" if gres else ""

    # Mail line
    mail_user = common.get("mail_user", "")
    mail_line = (
        f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={mail_user}"
        if mail_user
        else ""
    )

    # GPU check
    gpu_check = (
        'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv'
        if gres
        else "# No GPU allocated"
    )

    replacements = {
        "{{JOB_NAME}}": log_prefix,
        "{{LOG_PREFIX}}": log_prefix,
        "{{TIME}}": job.get("time", "02:00:00"),
        "{{PARTITION}}": profile.get("partition", "norm"),
        "{{MEM}}": profile.get("mem", "64G"),
        "{{CPUS}}": str(profile.get("cpus", 8)),
        "{{GRES_LINE}}": gres_line,
        "{{MAIL_LINE}}": mail_line,
        "{{DISPLAY_NAME}}": job.get("name", job_id),
        "{{MODULE_LINES}}": module_lines,
        "{{CONDA_ENV}}": common.get("conda_env", "secactpy"),
        "{{PROJECT_DIR}}": common.get("project_dir", "/data/parks34/projects/2cytoatlas"),
        "{{SCRIPT}}": job.get("script", ""),
        "{{ARGS}}": args,
        "{{GPU_CHECK}}": gpu_check,
    }

    script = template
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)

    return script


def submit_script(script_content: str, dependency_ids: list[str] | None = None) -> str:
    """Write script to a temp file and submit via sbatch. Returns job ID."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, dir="/tmp"
    ) as f:
        f.write(script_content)
        f.flush()
        script_path = f.name

    cmd = ["sbatch", "--parsable"]
    if dependency_ids:
        dep_str = ":".join(dependency_ids)
        cmd.append(f"--dependency=afterok:{dep_str}")
    cmd.append(script_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(script_path)

    if result.returncode != 0:
        print(f"  ERROR: sbatch failed: {result.stderr.strip()}", file=sys.stderr)
        return ""

    job_id = result.stdout.strip()
    return job_id


def expand_jobs(
    job_id: str, job: dict, profile: dict, common: dict
) -> list[tuple[str, str]]:
    """Expand a job definition into (label, script) pairs.

    Array jobs produce multiple pairs; simple jobs produce one.
    """
    array_items = job.get("array")
    if array_items:
        return [
            (f"{job_id}[{item}]", render_job_script(job_id, job, profile, common, item))
            for item in array_items
        ]
    return [(job_id, render_job_script(job_id, job, profile, common))]


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs from jobs.yaml"
    )
    parser.add_argument("--list", action="store_true", help="List available jobs")
    parser.add_argument("--dry-run", action="store_true", help="Show scripts without submitting")
    parser.add_argument("--jobs", type=str, help="Comma-separated job IDs to submit")
    parser.add_argument("--phases", type=str, help="Phase range to submit (e.g., 0-3)")
    parser.add_argument("--all", action="store_true", help="Submit all jobs")
    parser.add_argument("--no-deps", action="store_true", help="Ignore dependencies")

    args = parser.parse_args()

    config = load_config()
    profiles = config["profiles"]
    common = config["common"]
    jobs = config["jobs"]

    # --- List mode ---
    if args.list:
        print(f"{'ID':<30} {'Name':<35} {'Profile':<12} {'Time':<10} {'Phase'}")
        print("-" * 95)
        for job_id, job in sorted(jobs.items(), key=lambda x: x[1].get("phase", 0)):
            array_info = f" (x{len(job['array'])})" if job.get("array") else ""
            deps = ", ".join(job.get("depends_on", []))
            print(
                f"{job_id:<30} {job['name']:<35} {job['profile']:<12} "
                f"{job.get('time', 'N/A'):<10} {job.get('phase', 0)}"
                f"{array_info}"
            )
            if deps:
                print(f"{'':>30} depends: {deps}")
        return

    # --- Determine which jobs to submit ---
    if args.all:
        selected = set(jobs.keys())
    elif args.jobs:
        selected = set(args.jobs.split(","))
        unknown = selected - set(jobs.keys())
        if unknown:
            print(f"Unknown jobs: {unknown}", file=sys.stderr)
            sys.exit(1)
    elif args.phases:
        # Parse phase range
        if "-" in args.phases:
            lo, hi = args.phases.split("-", 1)
            phase_range = range(int(lo), int(hi) + 1)
        else:
            phase_range = [int(args.phases)]
        selected = {
            jid for jid, j in jobs.items()
            if j.get("phase", 0) in phase_range
        }
    else:
        parser.print_help()
        return

    # --- Topological sort by dependencies ---
    ordered = []
    visited = set()

    def visit(jid):
        if jid in visited or jid not in selected:
            return
        visited.add(jid)
        for dep in jobs[jid].get("depends_on", []):
            if dep in selected:
                visit(dep)
        ordered.append(jid)

    for jid in selected:
        visit(jid)

    # --- Submit ---
    print("=" * 60)
    print("CytoAtlas SLURM Job Submission")
    print(f"Jobs: {len(ordered)} selected")
    print("=" * 60)

    # Track submitted job IDs for dependency resolution
    submitted: dict[str, list[str]] = {}  # job_id -> list of SLURM job IDs

    for job_id in ordered:
        job = jobs[job_id]
        profile = profiles[job["profile"]]

        # Resolve dependencies
        dep_ids: list[str] = []
        if not args.no_deps:
            for dep in job.get("depends_on", []):
                if dep in submitted:
                    dep_ids.extend(submitted[dep])

        # Expand array jobs
        pairs = expand_jobs(job_id, job, profile, common)

        slurm_ids = []
        for label, script in pairs:
            if args.dry_run:
                print(f"\n--- {label} ---")
                print(script[:500])
                if len(script) > 500:
                    print(f"  ... ({len(script)} chars total)")
                slurm_ids.append("DRY_RUN")
            else:
                sid = submit_script(script, dep_ids if dep_ids else None)
                if sid:
                    slurm_ids.append(sid)
                    dep_str = f" (after {','.join(dep_ids)})" if dep_ids else ""
                    print(f"  Submitted {label}: {sid}{dep_str}")
                else:
                    print(f"  FAILED: {label}")

        submitted[job_id] = slurm_ids

    if not args.dry_run:
        print()
        print("=" * 60)
        total = sum(len(ids) for ids in submitted.values())
        print(f"Submitted {total} SLURM jobs")
        print("=" * 60)
        print()
        print("Monitor:  squeue -u $USER")
        print("Logs:     tail -f logs/*.out")


if __name__ == "__main__":
    main()
