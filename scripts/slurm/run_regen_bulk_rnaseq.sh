#!/bin/bash
#SBATCH --job-name=regen_bulk_rnaseq
#SBATCH --output=/data/parks34/projects/2secactpy/logs/regen_bulk_rnaseq_%j.out
#SBATCH --error=/data/parks34/projects/2secactpy/logs/regen_bulk_rnaseq_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2secactpy

python3 -c "
import sys
sys.path.insert(0, '/vf/users/parks34/projects/2secactpy/scripts')
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader('preprocess', '/vf/users/parks34/projects/2secactpy/scripts/14_preprocess_bulk_validation.py').load_module()

import json

print('Regenerating bulk_rnaseq_validation.json with all CytoSig + SecAct targets...')
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
        if ds in bulk_rnaseq:
            for st in bulk_rnaseq[ds].get('donor_scatter', {}):
                n = len(bulk_rnaseq[ds]['donor_scatter'][st])
                print(f'  {ds}/{st}: {n} targets')
else:
    print('ERROR: No data generated')
    sys.exit(1)

print('Done!')
"
