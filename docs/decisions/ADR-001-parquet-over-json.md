# ADR-001: Use Parquet for Large Data Files Instead of JSON

## Status
Accepted (implementation planned for Round 3)

## Context

The CytoAtlas API currently stores all visualization data in JSON format:
- 30+ JSON files (~500MB total)
- Largest files: validation data (175-336MB per atlas)
- Current behavior: Entire JSON file loaded into memory when requested

**Problem**: Large JSON files (>500MB) cause memory overhead:
- validation_cima.json: 336MB → fully loaded into memory
- validation_inflammation.json: 280MB → fully loaded into memory
- Validation data contains 100K+ records with many fields only some endpoints need

**Current bottleneck**:
```
GET /api/v1/validation/sample-level/cima/SecAct
→ Load full validation_cima.json (336MB) into memory
→ Filter by signature type & validation level
→ Return subset (1-2MB)
```

This creates unnecessary memory pressure, slower startup time, and increased response latency.

## Decision

Migrate large JSON files (>200MB) to **Apache Parquet** format with the following strategy:

1. **Immediate** (Round 2): Keep JSON for all files, implement repository pattern as abstraction
2. **Round 3**: Migrate largest files (validation_*.json) to Parquet
3. **Future**: Optional migration of other large datasets as performance bottleneck arises

**Implementation approach**:
- Use PyArrow as the Parquet library (Python integration with pandas)
- Keep JSON for smaller files (<100MB) for simplicity
- Use repository pattern to switch implementations transparently
- Maintain backward compatibility: API responses unchanged

**Storage layout**:
```
visualization/data/
├── validation/
│  ├── cima.parquet          # validation_cima.json migrated
│  ├── inflammation.parquet  # validation_inflammation.json migrated
│  └── scatlas.parquet       # validation_scatlas.json migrated
├── cima_correlations.json   # Keep as JSON (30MB)
├── disease_activity.json    # Keep as JSON (45MB)
└── ...                      # Other JSON files unchanged
```

## Consequences

### Positive
- **Memory efficiency**: Load only required columns/rows (60-80% reduction)
- **Query performance**: PyArrow predicate pushdown filters at disk level
- **Faster startup**: No need to load 300MB+ files into memory at initialization
- **Scalability**: Can handle 1GB+ files without memory issues
- **Backward compatible**: Repository pattern makes switch transparent to API consumers

### Negative
- **Additional dependency**: PyArrow adds to requirements
- **Complexity increase**: Different backends require testing
- **One-time migration effort**: Converting all validation data to Parquet
- **Tools**: JSON files are human-readable; Parquet requires tools to inspect
- **Version pinning**: PyArrow compatibility across Python versions

## Alternatives Considered

### Alternative A: Keep JSON, Implement Tiered Caching (Rejected)
- **Pros**: No new dependencies, no migration needed
- **Cons**: Still loads 336MB into memory, just keeps it longer
- **Why rejected**: Doesn't solve fundamental memory overhead

### Alternative B: Split Large JSON Files (Rejected)
- **Pros**: Simpler than Parquet, no new dependencies
- **Cons**: Manual sharding, increased endpoint complexity
- **Why rejected**: Parquet provides better query semantics

### Alternative C: PostgreSQL Backend (Rejected for now)
- **Pros**: ACID guarantees, full-featured querying
- **Cons**: Adds external dependency (database server), increased deployment complexity
- **Why rejected**: Parquet works for current scale; DB can be future upgrade path

## Related ADRs
- [ADR-002: Repository Pattern for Data Access](ADR-002-repository-pattern.md) - Enables this decision
- [ADR-003: RBAC Model](ADR-003-rbac-model.md) - Security layer can leverage repository pattern

## Implementation Notes

### Phase 1: Repository Pattern (Round 3, Week 1)
```python
# Define repository protocol
class DataRepository(Protocol):
    async def get_sample_level_validations(self, atlas: str) -> List[ValidationRecord]: ...
    async def get_validations_by_signature(self, atlas: str, signature: str) -> List[ValidationRecord]: ...

# JSON implementation (current)
class JSONValidationRepository(DataRepository): ...

# Parquet implementation (new)
class ParquetValidationRepository(DataRepository):
    def __init__(self, path: str):
        self.table = pa.parquet.read_table(path)

    async def get_sample_level_validations(self, atlas: str):
        # Use PyArrow predicate pushdown
        filtered = self.table.filter(pc.equal(pc.field('validation_type'), 'sample_level'))
        return filtered.to_pandas().to_dict('records')
```

### Phase 2: Migration (Round 3, Week 2)
```bash
# Convert large JSON files to Parquet
python scripts/migrate_to_parquet.py \
  --input visualization/data/validation_cima.json \
  --output visualization/data/validation/cima.parquet
```

### Phase 3: Testing & Rollout (Round 3, Week 3)
- A/B testing (JSON vs Parquet backends)
- Performance benchmarking (memory, latency, throughput)
- Rollout to production

## Timeline
- **Round 3 Sprint 1**: Repository pattern scaffolding
- **Round 3 Sprint 2**: Parquet migration & testing
- **Round 3 Sprint 3**: Performance validation & production deployment
