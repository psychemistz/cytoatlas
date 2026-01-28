#!/bin/bash
# Visualization Health Check Script
#
# Validates that the CytoAtlas visualization builds and functions correctly.
# Run this after making changes to verify everything works.
#
# Usage: ./check_viz_health.sh

set -e

PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
VIZ_DIR="$PROJECT_DIR/visualization"

echo "=============================================="
echo "  CytoAtlas Visualization Health Check"
echo "=============================================="
echo ""

# Track overall status
ERRORS=0
WARNINGS=0

# Function to report status
check_pass() {
    echo "  [PASS] $1"
}

check_fail() {
    echo "  [FAIL] $1"
    ((ERRORS++))
}

check_warn() {
    echo "  [WARN] $1"
    ((WARNINGS++))
}

# 1. Check directory structure
echo "1. Checking directory structure..."
[ -d "$VIZ_DIR" ] && check_pass "visualization/ directory exists" || check_fail "visualization/ directory missing"
[ -d "$VIZ_DIR/data" ] && check_pass "visualization/data/ directory exists" || check_warn "visualization/data/ directory missing"
[ -d "$VIZ_DIR/panels" ] && check_pass "visualization/panels/ directory exists" || check_warn "visualization/panels/ directory missing"

# 2. Check main files
echo ""
echo "2. Checking main files..."
[ -f "$VIZ_DIR/index.html" ] && check_pass "index.html exists" || check_fail "index.html missing"
[ -f "$VIZ_DIR/index_standalone.html" ] && check_pass "index_standalone.html exists" || check_warn "index_standalone.html missing"
[ -f "$VIZ_DIR/dev-server.py" ] && check_pass "dev-server.py exists" || check_fail "dev-server.py missing"

# 3. Check HTML validity
echo ""
echo "3. Checking HTML validity..."
python3 << 'EOF'
import sys
from html.parser import HTMLParser

class ValidationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.errors = []
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        # Self-closing tags don't need closing
        if tag not in ['br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr']:
            self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        elif tag in self.tag_stack:
            self.errors.append(f"Mismatched tag: </{tag}>")

try:
    with open('/vf/users/parks34/projects/2secactpy/visualization/index.html') as f:
        parser = ValidationParser()
        parser.feed(f.read())
        if parser.errors:
            for err in parser.errors[:5]:
                print(f"  HTML Error: {err}", file=sys.stderr)
            sys.exit(1)
    print("  [PASS] HTML syntax valid")
except FileNotFoundError:
    print("  [SKIP] index.html not found")
except Exception as e:
    print(f"  [FAIL] HTML validation error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

# 4. Check embedded data
echo ""
echo "4. Checking data files..."
if [ -f "$VIZ_DIR/data/embedded_data.js" ]; then
    python3 << 'EOF'
try:
    with open('/vf/users/parks34/projects/2secactpy/visualization/data/embedded_data.js') as f:
        content = f.read()
        if 'EMBEDDED_DATA' in content or 'const ' in content:
            print("  [PASS] Embedded data file has expected structure")
        else:
            print("  [WARN] Embedded data file may be malformed")
except Exception as e:
    print(f"  [FAIL] Error reading embedded data: {e}")
EOF
else
    check_warn "embedded_data.js not found (may not be generated yet)"
fi

# Count JSON files
JSON_COUNT=$(find "$VIZ_DIR/data" -name "*.json" 2>/dev/null | wc -l)
echo "  Found $JSON_COUNT JSON data files"

# 5. Check JavaScript syntax
echo ""
echo "5. Checking JavaScript files..."
if command -v node > /dev/null 2>&1; then
    for js_file in "$VIZ_DIR"/*.js "$VIZ_DIR"/data/*.js; do
        if [ -f "$js_file" ]; then
            if node --check "$js_file" 2>/dev/null; then
                check_pass "$(basename "$js_file") syntax valid"
            else
                check_fail "$(basename "$js_file") has syntax errors"
            fi
        fi
    done
else
    check_warn "Node.js not available for JavaScript validation"
fi

# 6. Check development server
echo ""
echo "6. Checking development server..."
python3 << 'EOF'
import sys
try:
    import http.server
    import socketserver
    print("  [PASS] Required server modules available")
except ImportError as e:
    print(f"  [FAIL] Missing module: {e}")
    sys.exit(1)
EOF

# 7. Check agent files
echo ""
echo "7. Checking agent configuration..."
AGENT_DIR="$PROJECT_DIR/agents"
AGENT_COUNT=$(find "$AGENT_DIR" -name "*.md" 2>/dev/null | wc -l)
if [ "$AGENT_COUNT" -ge 9 ]; then
    check_pass "All 9 agent prompt files present ($AGENT_COUNT found)"
else
    check_warn "Expected 9 agent files, found $AGENT_COUNT"
fi

# 8. Check DECISIONS.md
echo ""
echo "8. Checking decision log..."
[ -f "$PROJECT_DIR/DECISIONS.md" ] && check_pass "DECISIONS.md exists" || check_warn "DECISIONS.md missing"

# Summary
echo ""
echo "=============================================="
echo "  Health Check Summary"
echo "=============================================="
echo "  Errors:   $ERRORS"
echo "  Warnings: $WARNINGS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "  Status: UNHEALTHY - Please fix errors above"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo "  Status: DEGRADED - Consider addressing warnings"
    exit 0
else
    echo "  Status: HEALTHY"
    exit 0
fi
