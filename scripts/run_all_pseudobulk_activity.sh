#!/bin/bash
# Run activity inference on all pseudobulk files

set -e
cd /vf/users/parks34/projects/2cytoatlas
source ~/bin/myconda
conda activate secactpy

SCRIPT="scripts/09_atlas_validation_activity.py"

echo "=========================================="
echo "Pseudobulk Activity Inference - All Files"
echo "=========================================="
echo "Start: $(date)"
echo ""

# Find all pseudobulk files (excluding resampled)
for pb_file in $(find results/atlas_validation -name "*pseudobulk*.h5ad" ! -name "*resampled*" | sort); do
    # Get output directory (replace pseudobulk with activity in path)
    atlas_dir=$(dirname $(dirname "$pb_file"))
    basename=$(basename "$pb_file" .h5ad | sed 's/_pseudobulk//')
    
    # Determine output directory based on file type
    if [[ "$basename" == *"_donor"* ]]; then
        output_dir="${atlas_dir}/activity_donor"
    else
        output_dir="${atlas_dir}/activity"
    fi
    
    # Check if all 3 activity files exist
    all_exist=true
    for sig in cytosig lincytosig secact; do
        if [ ! -f "${output_dir}/${basename}_${sig}.h5ad" ]; then
            all_exist=false
            break
        fi
    done
    
    if [ "$all_exist" = true ]; then
        echo "SKIP: ${basename} (all activity files exist)"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing: $pb_file"
    echo "Output: $output_dir"
    echo "Time: $(date)"
    echo "=========================================="
    
    python "$SCRIPT" --input "$pb_file" --output-dir "$output_dir" --validate 2>&1
done

echo ""
echo "=========================================="
echo "ALL COMPLETE: $(date)"
echo "=========================================="
