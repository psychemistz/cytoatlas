#!/bin/bash
# Post-decision hook: Create GitHub issue for critical decisions
#
# This hook is called after a critical decision is logged to DECISIONS.md
# It creates a GitHub issue for decisions marked as requiring review.
#
# Usage: ./post-decision.sh "Decision Title" "priority"
#   priority: "critical" | "high" | "medium" | "low"

DECISION="$1"
PRIORITY="${2:-medium}"
PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
DECISIONS_FILE="$PROJECT_DIR/DECISIONS.md"
REPO="psychemistz/cytoatlas"

# Only create issues for critical or high priority decisions
if [ "$PRIORITY" = "critical" ] || [ "$PRIORITY" = "high" ]; then
    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        echo "Warning: GitHub CLI (gh) not found. Skipping issue creation."
        exit 0
    fi

    # Check if authenticated
    if ! gh auth status &> /dev/null; then
        echo "Warning: Not authenticated with GitHub. Skipping issue creation."
        exit 0
    fi

    # Get the last 50 lines of DECISIONS.md for context
    CONTEXT=$(tail -50 "$DECISIONS_FILE" 2>/dev/null || echo "Unable to read decisions file")

    # Determine labels based on priority
    LABELS="needs-review"
    if [ "$PRIORITY" = "critical" ]; then
        LABELS="$LABELS,critical"
    fi

    # Create the issue
    BODY="## Decision Requiring Review

**Decision:** $DECISION
**Priority:** $PRIORITY

## Context from DECISIONS.md

\`\`\`markdown
$CONTEXT
\`\`\`

---
*This issue was automatically created by the CytoAtlas development orchestrator.*
"

    # Create issue (uncomment when repo is set up)
    # gh issue create \
    #     --repo "$REPO" \
    #     --title "[Decision Review] $DECISION" \
    #     --body "$BODY" \
    #     --label "$LABELS"

    echo "GitHub issue would be created for: $DECISION (priority: $PRIORITY)"
    echo "Labels: $LABELS"
fi

exit 0
