#!/bin/bash
# Security audit script for CytoAtlas API
# Runs pip-audit for dependency vulnerabilities and bandit for code security issues

set -e

echo "=================================="
echo "CytoAtlas API Security Audit"
echo "=================================="
echo ""

echo "=== Running pip-audit ==="
echo "Checking for known vulnerabilities in dependencies..."
pip-audit --strict || {
    echo "⚠️  WARNING: Vulnerabilities found in dependencies"
    exit 1
}
echo "✓ No vulnerabilities found"
echo ""

echo "=== Running bandit ==="
echo "Scanning code for security issues..."
bandit -r app/ -ll -ii || {
    echo "⚠️  WARNING: Security issues found in code"
    exit 1
}
echo "✓ No security issues found"
echo ""

echo "=================================="
echo "Security audit completed successfully"
echo "=================================="
