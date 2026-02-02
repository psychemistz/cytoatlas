# handoff

Create a handoff document for session continuity

## Instructions

When the user invokes /handoff, create a session summary document:

1. **Gather context**:
   - Current git branch and recent commits
   - Files modified in this session
   - Tasks completed
   - Open issues or blockers

2. **Create handoff document** at `.claude/handoffs/YYYY-MM-DD-HHMMSS.md`:

```markdown
# Session Handoff - [Date]

## Summary
[Brief description of what was accomplished]

## Completed Tasks
- [ ] Task 1
- [ ] Task 2

## Files Modified
- `path/to/file1.py` - Description of changes
- `path/to/file2.json` - Description of changes

## Current State
- Branch: `branch-name`
- Last commit: `commit-hash` - message

## Next Steps
1. Step 1
2. Step 2

## Blockers/Issues
- Issue 1
- Issue 2

## Context for Next Session
[Any important context that should be known]
```

3. **Save and report** the handoff file location

## Arguments

- `/handoff` - Create full handoff document
- `/handoff quick` - Create abbreviated summary
