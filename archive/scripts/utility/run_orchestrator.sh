#!/bin/bash
# Launch multi-agent orchestration session for CytoAtlas development
#
# Usage:
#   ./run_orchestrator.sh                    # Start new session
#   ./run_orchestrator.sh --resume <id>      # Resume previous session
#   ./run_orchestrator.sh --task "message"   # Start with specific task
#
# Environment variables:
#   MAX_TURNS: Maximum number of turns (default: 200)
#   LOG_DIR: Directory for session logs

set -e

# Configuration
PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/agent_sessions}"
MAX_TURNS="${MAX_TURNS:-200}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="$LOG_DIR/session_$TIMESTAMP.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Parse arguments
RESUME_ID=""
TASK_MESSAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_ID="$2"
            shift 2
            ;;
        --task)
            TASK_MESSAGE="$2"
            shift 2
            ;;
        --help)
            echo "CytoAtlas Multi-Agent Orchestrator"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resume <id>     Resume a previous session"
            echo "  --task <message>  Start with a specific task message"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MAX_TURNS         Maximum number of turns (default: 200)"
            echo "  LOG_DIR           Directory for session logs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Export environment variables
export DECISIONS_LOG="$PROJECT_DIR/DECISIONS.md"
export SESSION_LOG

# Print banner
echo "=============================================="
echo "  CytoAtlas Multi-Agent Orchestrator"
echo "=============================================="
echo "  Project:     $PROJECT_DIR"
echo "  Session log: $SESSION_LOG"
echo "  Max turns:   $MAX_TURNS"
echo "=============================================="
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Build command
CMD="claude"

if [ -n "$RESUME_ID" ]; then
    echo "Resuming session: $RESUME_ID"
    CMD="$CMD --resume $RESUME_ID"
elif [ -n "$TASK_MESSAGE" ]; then
    echo "Starting with task: $TASK_MESSAGE"
    CMD="$CMD -p \"$TASK_MESSAGE\""
else
    # Default orchestrator task
    DEFAULT_TASK="Read the orchestrator agent prompt at agents/orchestrator.md and begin autonomous development of the CytoAtlas visualization panels. Check DECISIONS.md for context and the current TODO list. Log all critical decisions."
    CMD="$CMD -p \"$DEFAULT_TASK\""
fi

# Run Claude Code with logging
echo "Starting session at $(date)"
echo ""

eval "$CMD" 2>&1 | tee "$SESSION_LOG"

echo ""
echo "Session ended at $(date)"
echo "Log saved to: $SESSION_LOG"
