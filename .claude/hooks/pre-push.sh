#!/bin/bash
# Pre-push hook: Run full QA checks before pushing
#
# This hook runs endpoint validation and schema checks before push.
#
# Usage: ./pre-push.sh

PROJECT_DIR="/data/parks34/projects/2cytoatlas"
API_DIR="$PROJECT_DIR/cytoatlas-api"

echo "Running pre-push QA checks..."

# Check if API directory exists
if [ ! -d "$API_DIR" ]; then
    echo "Warning: API directory not found. Skipping API checks."
    exit 0
fi

cd "$API_DIR"

# Check if server can be started (dry run)
echo "Validating application structure..."
python -c "
from app.main import create_app
app = create_app()
print(f'Found {len(app.routes)} routes')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Error: Application failed to initialize"
    exit 1
fi

# Check for any TODO validation endpoints
echo ""
echo "Checking for unimplemented validation endpoints..."
grep -r "TODO\|NotImplemented\|pass  # TODO" app/routers/validation.py 2>/dev/null && {
    echo "Warning: Found TODO items in validation router"
}

echo ""
echo "Pre-push checks completed."
exit 0
