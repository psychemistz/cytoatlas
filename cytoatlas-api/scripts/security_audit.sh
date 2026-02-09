#!/bin/bash
# =============================================================================
# CytoAtlas API Security Audit Script
# =============================================================================
# Runs comprehensive security checks:
#   1. pip-audit  - known CVEs in dependencies
#   2. bandit     - static analysis for Python security issues
#   3. secret scan - detect hardcoded secrets in source code
#   4. summary report
#
# Usage:
#   ./scripts/security_audit.sh [--json] [--strict]
#
# Options:
#   --json     Output bandit results as JSON
#   --strict   Exit with non-zero on any finding
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_DIR="$PROJECT_DIR/app"
REPORT_DIR="$PROJECT_DIR/security"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/audit_report_${TIMESTAMP}.txt"

# Parse arguments
JSON_OUTPUT=false
STRICT_MODE=false
for arg in "$@"; do
    case $arg in
        --json) JSON_OUTPUT=true ;;
        --strict) STRICT_MODE=true ;;
    esac
done

# Ensure report directory exists
mkdir -p "$REPORT_DIR"

# Counters
TOTAL_ISSUES=0
CRITICAL_ISSUES=0

echo "==================================================="
echo " CytoAtlas API Security Audit"
echo " Date: $(date)"
echo " Project: $PROJECT_DIR"
echo "==================================================="
echo ""

# Redirect output to both console and report file
exec > >(tee -a "$REPORT_FILE") 2>&1

# ---- Phase 1: Dependency Vulnerability Scan ----
echo "--- Phase 1: Dependency Vulnerability Scan (pip-audit) ---"
echo ""

if command -v pip-audit &>/dev/null; then
    echo "Running pip-audit..."
    if pip-audit --strict 2>&1; then
        echo "[PASS] No known vulnerabilities found in dependencies"
    else
        echo "[WARN] Vulnerabilities found in dependencies"
        TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
    fi
else
    echo "[SKIP] pip-audit not installed (install with: pip install pip-audit)"
fi
echo ""

# ---- Phase 2: Static Analysis (Bandit) ----
echo "--- Phase 2: Static Security Analysis (bandit) ---"
echo ""

if command -v bandit &>/dev/null; then
    echo "Running bandit on $APP_DIR ..."

    if [ "$JSON_OUTPUT" = true ]; then
        BANDIT_OUT="$REPORT_DIR/bandit_results_${TIMESTAMP}.json"
        bandit -r "$APP_DIR" -f json -o "$BANDIT_OUT" -ll -ii 2>/dev/null || true
        echo "JSON results saved to: $BANDIT_OUT"
        # Count issues from JSON
        if command -v python3 &>/dev/null; then
            BANDIT_COUNT=$(python3 -c "
import json, sys
try:
    data = json.load(open('$BANDIT_OUT'))
    results = data.get('results', [])
    print(len(results))
except: print(0)
")
            echo "Found $BANDIT_COUNT issues"
            TOTAL_ISSUES=$((TOTAL_ISSUES + BANDIT_COUNT))
        fi
    else
        if bandit -r "$APP_DIR" -ll -ii 2>&1; then
            echo "[PASS] No security issues found by bandit"
        else
            echo "[WARN] Security issues found by bandit"
            TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
        fi
    fi
else
    echo "[SKIP] bandit not installed (install with: pip install bandit)"
fi
echo ""

# ---- Phase 3: Hardcoded Secrets Scan ----
echo "--- Phase 3: Hardcoded Secrets Scan ---"
echo ""

SECRET_PATTERNS=(
    'api_key\s*=\s*["\x27][^"\x27]{8,}'
    'password\s*=\s*["\x27][^"\x27]{4,}'
    'secret\s*=\s*["\x27][^"\x27]{8,}'
    'token\s*=\s*["\x27][^"\x27]{8,}'
    'ANTHROPIC_API_KEY\s*=\s*["\x27]sk-'
    'OPENAI_API_KEY\s*=\s*["\x27]sk-'
    'AWS_SECRET_ACCESS_KEY\s*='
    'private_key\s*=\s*["\x27]'
)

SECRET_FOUND=0
for pattern in "${SECRET_PATTERNS[@]}"; do
    # Search Python files, excluding __pycache__, .env, .git
    MATCHES=$(grep -rn --include="*.py" -E "$pattern" "$APP_DIR" 2>/dev/null \
        | grep -v "__pycache__" \
        | grep -v "test_" \
        | grep -v "\.env" \
        | grep -v "# " \
        | grep -v "Field(default=" \
        | grep -v "get_settings" \
        | grep -v "os.environ" \
        | grep -v "example" \
        || true)

    if [ -n "$MATCHES" ]; then
        echo "[WARN] Potential hardcoded secret (pattern: $pattern):"
        echo "$MATCHES" | head -5
        echo ""
        SECRET_FOUND=$((SECRET_FOUND + 1))
    fi
done

if [ "$SECRET_FOUND" -eq 0 ]; then
    echo "[PASS] No hardcoded secrets detected"
else
    echo "[WARN] $SECRET_FOUND potential hardcoded secret patterns found"
    TOTAL_ISSUES=$((TOTAL_ISSUES + SECRET_FOUND))
fi

# Also check .env files are not committed
echo ""
echo "Checking for .env files in git..."
if git -C "$PROJECT_DIR" ls-files --cached ".env" "*.env" 2>/dev/null | grep -q ".env"; then
    echo "[CRITICAL] .env file is tracked by git!"
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
else
    echo "[PASS] No .env files tracked by git"
fi
echo ""

# ---- Phase 4: Configuration Security Check ----
echo "--- Phase 4: Configuration Security Check ---"
echo ""

# Check for debug mode in any config
DEBUG_REFS=$(grep -rn --include="*.py" "debug\s*=\s*True" "$APP_DIR" 2>/dev/null \
    | grep -v "__pycache__" \
    | grep -v "test_" \
    | grep -v "Field(" \
    || true)

if [ -n "$DEBUG_REFS" ]; then
    echo "[WARN] Hardcoded debug=True found:"
    echo "$DEBUG_REFS"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
else
    echo "[PASS] No hardcoded debug=True found"
fi

# Check CORS wildcards
CORS_WILDCARDS=$(grep -rn --include="*.py" 'allow_origins=\["\*"\]' "$APP_DIR" 2>/dev/null \
    | grep -v "__pycache__" \
    || true)

if [ -n "$CORS_WILDCARDS" ]; then
    echo "[WARN] Wildcard CORS origin found:"
    echo "$CORS_WILDCARDS"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
else
    echo "[PASS] No wildcard CORS origins in code"
fi

echo ""

# ---- Summary ----
echo "==================================================="
echo " AUDIT SUMMARY"
echo "==================================================="
echo " Total issues found: $TOTAL_ISSUES"
echo " Critical issues:    $CRITICAL_ISSUES"
echo " Report saved to:    $REPORT_FILE"
echo "==================================================="

if [ "$STRICT_MODE" = true ] && [ "$TOTAL_ISSUES" -gt 0 ]; then
    echo ""
    echo "STRICT MODE: Exiting with error due to findings."
    exit 1
fi

if [ "$CRITICAL_ISSUES" -gt 0 ]; then
    echo ""
    echo "CRITICAL issues found - these must be addressed before deployment."
    exit 1
fi

echo ""
echo "Audit complete."
exit 0
