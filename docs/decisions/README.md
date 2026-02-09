# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting major design decisions in the CytoAtlas project. ADRs explain the context, rationale, and consequences of decisions to help future maintainers understand design trade-offs.

## Format

Each ADR follows this template:

```
# ADR-NNN: [Decision Title]

## Status
- Proposed, Accepted, Deprecated, Superseded

## Context
[Background information and constraints that led to this decision]

## Decision
[The chosen approach]

## Consequences
[Positive and negative outcomes of this decision]

## Alternatives Considered
[Other options evaluated and why they were rejected]

## Related ADRs
[Links to related decisions]
```

## ADRs

### [ADR-001: Parquet over JSON for Large Data Files](ADR-001-parquet-over-json.md)
**Status**: Accepted (for Round 3 implementation)

Use Parquet for large data files (validation_*.json >500MB) instead of JSON. Provides:
- Faster query performance (predicate pushdown)
- Better memory efficiency (columnar storage)
- Backward compatibility through repository pattern

### [ADR-002: Repository Pattern for Data Access](ADR-002-repository-pattern.md)
**Status**: Accepted (for Round 3 implementation)

Implement protocol-based repository abstraction for data access. Enables:
- Testability (mock repositories in unit tests)
- Backend swappability (JSON → Parquet → PostgreSQL)
- Separation of concerns (data access from business logic)

### [ADR-003: Role-Based Access Control Model](ADR-003-rbac-model.md)
**Status**: Accepted (for Round 2 implementation)

Implement 5-role RBAC model (anonymous, viewer, researcher, data_curator, admin). Provides:
- Granular permission control
- Clear escalation path for different user types
- Foundation for audit logging

## Decision Log

| ADR | Title | Status | Date | Round |
|-----|-------|--------|------|-------|
| 001 | Parquet over JSON | Accepted | 2026-02-09 | 3 |
| 002 | Repository Pattern | Accepted | 2026-02-09 | 3 |
| 003 | RBAC Model | Accepted | 2026-02-09 | 2 |

## Guidelines for New ADRs

1. **When to write an ADR**: For decisions that affect multiple components or have long-term architectural impact
2. **Format**: Use the template above
3. **Length**: Keep to 1-2 pages (concise but complete)
4. **Context**: Explain constraints and trade-offs, not just the decision
5. **Review**: Get team consensus before marking as "Accepted"
6. **Numbering**: Sequential (ADR-004, ADR-005, etc.)

## References

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture overview
- [CLAUDE.md](../../CLAUDE.md) - Current status and TODOs
- [Archive of stale plans](../archive/) - Historical planning documents

## ADR Template (Copy & Modify)

```markdown
# ADR-NNN: [Decision Title]

## Status
Proposed | Accepted | Deprecated | Superseded by ADR-XXX

## Context
[1-2 paragraphs explaining the problem, constraints, and background]

## Decision
[The chosen approach - 1-2 paragraphs]

## Consequences

### Positive
- [Benefit 1]
- [Benefit 2]

### Negative
- [Trade-off 1]
- [Trade-off 2]

## Alternatives Considered
- [Alternative A]: Why rejected
- [Alternative B]: Why rejected

## Related ADRs
- ADR-XXX (related topic)
```
