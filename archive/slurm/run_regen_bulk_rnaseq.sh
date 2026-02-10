#!/bin/bash
#SBATCH --job-name=regen_bulk_rnaseq
#SBATCH --output=/data/parks34/projects/2cytoatlas/logs/regen_bulk_rnaseq_%j.out
#SBATCH --error=/data/parks34/projects/2cytoatlas/logs/regen_bulk_rnaseq_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas

python3 -c "
import sys
sys.path.insert(0, '/vf/users/parks34/projects/2cytoatlas/scripts')
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader('preprocess', '/vf/users/parks34/projects/2cytoatlas/scripts/14_preprocess_bulk_validation.py').load_module()

import json

print('Regenerating bulk_rnaseq_validation.json (all points, no subsampling)...')
bulk_rnaseq = mod.build_bulk_rnaseq_json()

if bulk_rnaseq:
    output_path = mod.BULK_RNASEQ_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(bulk_rnaseq, f, separators=(',', ':'))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f'Output: {output_path}')
    print(f'Size: {size_mb:.1f} MB')

    # Print target counts
    for ds in ['gtex', 'tcga']:
        if ds not in bulk_rnaseq:
            continue
        d = bulk_rnaseq[ds]
        for section in ['donor_scatter', 'stratified_scatter']:
            if section not in d:
                continue
            data = d[section]
            if section == 'stratified_scatter':
                for level, sigs in data.items():
                    for st, targets in sigs.items():
                        if targets:
                            first_t = list(targets.values())[0]
                            n_pts = len(first_t.get('points', []))
                            print(f'  {ds}/{section}/{level}/{st}: {len(targets)} targets, {n_pts} pts/target')
            else:
                for st, targets in data.items():
                    if targets:
                        first_t = list(targets.values())[0]
                        n_pts = len(first_t.get('points', []))
                        print(f'  {ds}/{section}/{st}: {len(targets)} targets, {n_pts} pts/target')
else:
    print('ERROR: No data generated')
    sys.exit(1)

print('Done!')
"
