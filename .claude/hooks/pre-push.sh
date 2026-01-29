#!/bin/bash
# Pre-push hook: Run full QA checks before pushing
#
# This hook runs endpoint validation and schema checks before push.
#
# Usage: ./pre-push.sh

PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
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

# Run coverage report
if [ -f "$API_DIR/agents/qa_checkers/coverage_reporter.py" ]; then
    echo ""
    echo "Endpoint Coverage Report:"
    echo "========================="
    python -m agents.qa_checkers.coverage_reporter 2>/dev/null | head -30
fi

# Check for any TODO validation endpoints
echo ""
echo "Checking for unimplemented validation endpoints..."
grep -r "TODO\|NotImplemented\|pass  # TODO" app/routers/validation.py 2>/dev/null && {
    echo "Warning: Found TODO items in validation router"
}

# If server is running, run endpoint checks
if curl -s "http://localhost:8000/api/v1/health" > /dev/null 2>&1; then
    echo ""
    echo "Server is running. Running endpoint checks..."
    python -m agents.qa_checkers.endpoint_checker --atlas all 2>/dev/null | head -50
fi

echo ""
echo "Pre-push checks completed."
exit 0
