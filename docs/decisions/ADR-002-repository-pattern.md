# ADR-002: Use Repository Pattern for Data Access

## Status
Accepted (implementation planned for Round 3)

## Context

Currently, the CytoAtlas API services directly load and manipulate JSON data:

```python
# Current approach (tight coupling)
class CIMAService:
    def get_correlations(self, gene: str):
        # Load entire JSON file
        data = json.load('visualization/data/cima_correlations.json')
        # Filter in-memory
        return [item for item in data if item['gene'] == gene]
```

**Problems with current approach**:
1. **Tight coupling**: Services hardcoded to JSON format
2. **Poor testability**: Unit tests require actual JSON files
3. **Limited flexibility**: Switching backends (JSON → Parquet → PostgreSQL) requires refactoring services
4. **Duplication**: Multiple services duplicate JSON loading logic
5. **Hard to cache**: Cache logic scattered across services

**Real-world impact**:
- If we want to migrate to Parquet (ADR-001), we must refactor ~12 services
- Unit tests fail if JSON files missing
- Hard to add caching layer without service modifications

## Decision

Implement **Repository Pattern** with protocol-based abstraction to separate data access from business logic:

```python
# Protocol definition (interface)
class DataRepository(Protocol):
    """Abstract data access interface"""
    async def get_correlations(self, gene: str) -> List[CorrelationData]: ...
    async def get_disease_activity(self, disease: str) -> List[ActivityData]: ...

# JSON implementation (current)
class JSONRepository(DataRepository):
    async def get_correlations(self, gene: str):
        data = json.load('visualization/data/cima_correlations.json')
        return [item for item in data if item['gene'] == gene]

# Parquet implementation (future)
class ParquetRepository(DataRepository):
    async def get_correlations(self, gene: str):
        table = pa.parquet.read_table('visualization/data/correlations.parquet')
        filtered = table.filter(pc.equal(pc.field('gene'), gene))
        return filtered.to_pandas().to_dict('records')

# PostgreSQL implementation (future)
class PostgreSQLRepository(DataRepository):
    async def get_correlations(self, gene: str):
        return await db.execute(
            "SELECT * FROM correlations WHERE gene = :gene",
            {"gene": gene}
        )

# Service layer (decoupled from data storage)
class CIMAService:
    def __init__(self, repo: DataRepository):
        self.repo = repo  # Injected dependency

    async def get_correlations(self, gene: str):
        return await self.repo.get_correlations(gene)

# Dependency injection
json_repo = JSONRepository()
service = CIMAService(json_repo)

# Easy to switch at runtime
parquet_repo = ParquetRepository()
service_v2 = CIMAService(parquet_repo)  # Same service, different backend!
```

## Consequences

### Positive
- **Testability**: Mock repository in unit tests (no file I/O needed)
- **Backend flexibility**: Switch implementations without changing services
- **Reduced duplication**: Shared repository base class
- **Clear separation**: Data access logic separated from business logic
- **DDD alignment**: Repository pattern core to domain-driven design
- **Future-proof**: Enables migration to Parquet, PostgreSQL, or other backends

### Negative
- **Additional abstraction**: One more layer to understand
- **Slight overhead**: Protocol dispatch (negligible in Python)
- **Migration cost**: Refactor 12 services (~500 lines of code)
- **Testing complexity**: Need tests for each repository implementation
- **Requires planning**: Must define repository interface upfront

## Alternatives Considered

### Alternative A: Adapter Pattern (Rejected)
- **Pros**: Minimal changes to existing services
- **Cons**: Still requires wrapper layer, less clean abstraction
- **Why rejected**: Repository pattern more idiomatic for data access

### Alternative B: Dependency Injection without Protocols (Rejected)
- **Pros**: Works in runtime-typed Python
- **Cons**: Less type safety, harder to document expected interface
- **Why rejected**: Protocols provide better code clarity

### Alternative C: Direct Database-Backed API (Rejected)
- **Pros**: Single source of truth, ACID guarantees
- **Cons**: PostgreSQL adds deployment complexity, requires database administration
- **Why rejected**: JSON/Parquet sufficient for current scale; DB later if needed

## Related ADRs
- [ADR-001: Parquet over JSON](ADR-001-parquet-over-json.md) - Enabled by this decision
- [ADR-003: RBAC Model](ADR-003-rbac-model.md) - Can leverage repository for permission checking

## Implementation Notes

### Step 1: Define Repository Protocol

```python
# cytoatlas-api/app/repositories/base.py
from typing import Protocol, List, Any

class CorrelationData(TypedDict):
    gene: str
    protein: str
    rho: float
    p_value: float
    count: int

class DataRepository(Protocol):
    """Protocol for data access implementations"""

    async def get_correlations(
        self, gene: str, limit: int = 100
    ) -> List[CorrelationData]: ...

    async def get_disease_activity(
        self, disease: str, signature: str
    ) -> List[ActivityData]: ...

    async def get_validation_data(
        self, atlas: str, validation_type: str
    ) -> List[ValidationData]: ...
```

### Step 2: Create Repository Implementations

```python
# cytoatlas-api/app/repositories/json.py
class JSONRepository(DataRepository):
    def __init__(self, data_path: str):
        self.data_path = data_path

    async def get_correlations(self, gene: str, limit: int = 100):
        # Load and filter JSON
        ...

# cytoatlas-api/app/repositories/parquet.py (future)
class ParquetRepository(DataRepository):
    def __init__(self, data_path: str):
        self.data_path = data_path

    async def get_correlations(self, gene: str, limit: int = 100):
        # Load and filter Parquet
        ...
```

### Step 3: Inject Repository into Services

```python
# cytoatlas-api/app/services/cima_service.py
class CIMAService:
    def __init__(self, repo: DataRepository):
        self.repo = repo

    async def correlations(self, gene: str):
        return await self.repo.get_correlations(gene)
```

### Step 4: Unit Tests (Example)

```python
# tests/test_cima_service.py
class MockRepository(DataRepository):
    async def get_correlations(self, gene: str, limit: int = 100):
        return [
            {
                'gene': gene,
                'protein': 'IL17A',
                'rho': 0.85,
                'p_value': 0.001,
                'count': 100
            }
        ]

def test_cima_service():
    mock_repo = MockRepository()
    service = CIMAService(mock_repo)
    result = await service.correlations('TNF')
    assert len(result) == 1
    assert result[0]['rho'] == 0.85
```

## Migration Timeline

- **Round 3 Week 1**: Define repository protocols
- **Round 3 Week 2**: Implement JSON repository, refactor services
- **Round 3 Week 3**: Implement Parquet repository, add tests
- **Round 4**: PostgreSQL repository (optional upgrade path)

## Files to Modify

- `cytoatlas-api/app/repositories/` (create new directory)
  - `__init__.py`
  - `base.py` (protocols)
  - `json.py` (JSON implementation)
  - `parquet.py` (Parquet implementation, future)
  - `postgresql.py` (PostgreSQL implementation, future)

- `cytoatlas-api/app/services/` (refactor existing ~12 services)
  - Inject repository in `__init__`
  - Replace direct JSON loading with repository calls

- `tests/` (add repository tests)
  - `test_repositories.py`
  - Update service tests to use mock repositories
