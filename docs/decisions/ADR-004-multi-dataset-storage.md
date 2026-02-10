# ADR-004: Separate DuckDB Files per Dataset Domain

## Status
Accepted

## Context

CytoAtlas is expanding from 3 atlases (~19M cells, single `atlas_data.duckdb` at 590 MB) to 6 datasets (~240M cells) with the addition of:

1. **parse_10M** — 9.7M cytokine-perturbed PBMCs (perturbation domain)
2. **Tahoe-100M** — 100.6M drug-perturbed cancer cells (perturbation domain)
3. **SpatialCorpus-110M** — 110M spatial transcriptomics cells (spatial domain)

These new datasets represent fundamentally different bounded contexts (Perturbation Response, Drug Sensitivity, Spatial Biology) with distinct data schemas, analysis pipelines, and lifecycle requirements.

**Problem**: Expanding the single `atlas_data.duckdb` to accommodate all datasets would:
- Create a single point of failure for all data access
- Require regenerating the entire database when any domain changes
- Mix unrelated schemas (observational atlas, perturbation, spatial) in one file
- Risk regressions in existing API endpoints during data updates
- Result in a monolithic database (~8-10 GB) harder to manage on HPC

## Decision

Use **separate DuckDB files per dataset domain**:

| Database | Contents | Estimated Size |
|----------|----------|----------------|
| `atlas_data.duckdb` (existing) | CIMA, Inflammation, scAtlas (19M cells) | 590 MB |
| `perturbation_data.duckdb` (new) | parse_10M + Tahoe-100M aggregated results | ~3-5 GB |
| `spatial_data.duckdb` (new) | SpatialCorpus-110M aggregated results | ~2-4 GB |

Each database is **fully independent** — no cross-database `ATTACH` queries.

The DuckDB repository layer (`duckdb_repository.py`) is extended with:
- `_DATABASE_REGISTRY`: Maps domain names to database file paths
- `_TABLE_TO_DATABASE`: Maps each table name to its owning database
- Lazy connection initialization per database (same read-only pattern)

## Consequences

### Positive
- **Zero regression risk**: Existing `atlas_data.duckdb` is untouched
- **Independent lifecycle**: Each database can be regenerated without affecting others
- **Parallelizable**: 3 conversion scripts can run simultaneously
- **Domain alignment**: Database boundaries match bounded contexts
- **Simpler debugging**: Issues isolated to specific domain databases
- **Faster regeneration**: Smaller files regenerate faster

### Negative
- **Multiple connections**: Repository manages 3 DuckDB connections instead of 1
- **No cross-domain joins**: Cannot JOIN perturbation and spatial tables directly
- **Configuration complexity**: 3 file paths to manage instead of 1
- **Disk overhead**: Slight duplication of metadata across databases

## Alternatives Considered

### Alternative A: Expand Single DuckDB (Rejected)
- **Pros**: Simpler connection management, cross-domain JOINs possible
- **Cons**: Single point of failure, monolithic regeneration, regression risk
- **Why rejected**: Violates bounded context separation; too risky for existing endpoints

### Alternative B: DuckDB ATTACH for Cross-Queries (Rejected)
- **Pros**: Separate files with cross-database query capability
- **Cons**: ATTACH adds complexity, connection management harder, potential locking issues
- **Why rejected**: No current use case for cross-domain JOINs; can add ATTACH later if needed

### Alternative C: PostgreSQL Migration (Rejected for now)
- **Pros**: Full RDBMS features, concurrent writes, ACID
- **Cons**: External dependency, deployment complexity on HPC
- **Why rejected**: DuckDB works well for read-heavy analytics; PostgreSQL overkill for current scale

## Related ADRs
- [ADR-001: Parquet over JSON](ADR-001-parquet-over-json.md) — Addressed by DuckDB migration
- [ADR-002: Repository Pattern](ADR-002-repository-pattern.md) — Repository abstraction enables multi-database support

## Implementation Notes

### Repository Layer Changes

```python
# duckdb_repository.py
_DATABASE_REGISTRY = {
    'atlas': 'atlas_data.duckdb',
    'perturbation': 'perturbation_data.duckdb',
    'spatial': 'spatial_data.duckdb',
}

_TABLE_TO_DATABASE = {
    'parse10m_activity': 'perturbation',
    'tahoe_activity': 'perturbation',
    'spatial_activity': 'spatial',
    # ... all new tables mapped to their domain
    # Existing tables default to 'atlas'
}
```

### Database Generation

```bash
# Independent generation scripts
python scripts/convert_data_to_duckdb.py --all       # atlas_data.duckdb (existing)
python scripts/convert_perturbation_to_duckdb.py      # perturbation_data.duckdb
python scripts/convert_spatial_to_duckdb.py            # spatial_data.duckdb
```

### Verification

```bash
# Check each database independently
python -c "import duckdb; db=duckdb.connect('atlas_data.duckdb', read_only=True); print(db.execute('SHOW TABLES').fetchall())"
python -c "import duckdb; db=duckdb.connect('perturbation_data.duckdb', read_only=True); print(db.execute('SHOW TABLES').fetchall())"
python -c "import duckdb; db=duckdb.connect('spatial_data.duckdb', read_only=True); print(db.execute('SHOW TABLES').fetchall())"
```
