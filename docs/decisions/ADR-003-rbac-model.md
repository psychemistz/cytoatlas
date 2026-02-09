# ADR-003: Implement Role-Based Access Control (RBAC) Model

## Status
Accepted (implementation planned for Round 2)

## Context

Currently, the CytoAtlas API has **minimal authentication**:
- `core/security.py` contains JWT scaffolding (not enforced)
- User model has single `is_admin` boolean flag
- No permission checks on endpoints
- All data accessible to anyone (anonymous users)

**Problems with current approach**:
1. **One-dimensional access**: Only `admin` vs. `non-admin`
2. **No audit trail**: Can't track who accessed what data
3. **Scalability limitation**: Hard to add fine-grained permissions later
4. **Compliance risk**: Cannot enforce data access restrictions
5. **No escalation path**: No way to grant intermediate permissions

**Real-world scenarios not supported**:
- Researcher needs data export but cannot administer users
- Data curator should manage metadata but cannot access admin panel
- Public user should see dashboard but not download data
- Audit trail needed for grant compliance (NIH requirements)

## Decision

Implement a **five-role Role-Based Access Control (RBAC) model** with explicit permission checking:

### Role Hierarchy

```
┌────────────────────────────────────────┐
│ admin (system administration)           │
│  └─ Full access to all operations      │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ data_curator (dataset management)       │
│  ├─ Submit/edit datasets                │
│  ├─ Manage metadata                     │
│  └─ Access researcher features          │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ researcher (data access)                │
│  ├─ Download CSV/JSON exports           │
│  ├─ Access advanced analytics           │
│  └─ Access viewer features              │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ viewer (dashboard access)               │
│  ├─ View all public visualizations      │
│  ├─ Search signatures/cell types        │
│  └─ Access anonymous features           │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ anonymous (public access)               │
│  ├─ View public dashboards              │
│  └─ Basic search (limited results)      │
└────────────────────────────────────────┘
```

### Permission Matrix

| Operation | Anonymous | Viewer | Researcher | Data Curator | Admin |
|-----------|-----------|--------|-----------|--------------|-------|
| View dashboard | ✅ | ✅ | ✅ | ✅ | ✅ |
| Search (unlimited) | ❌ | ✅ | ✅ | ✅ | ✅ |
| Download data (CSV/JSON) | ❌ | ❌ | ✅ | ✅ | ✅ |
| Submit dataset | ❌ | ❌ | ❌ | ✅ | ✅ |
| Manage metadata | ❌ | ❌ | ❌ | ✅ | ✅ |
| Manage users | ❌ | ❌ | ❌ | ❌ | ✅ |
| View audit logs | ❌ | ❌ | ❌ | ❌ | ✅ |
| System configuration | ❌ | ❌ | ❌ | ❌ | ✅ |

### Implementation

```python
# cytoatlas-api/app/models/user.py
from enum import Enum

class Role(str, Enum):
    ANONYMOUS = "anonymous"      # Unauthenticated
    VIEWER = "viewer"            # Dashboard access
    RESEARCHER = "researcher"    # Data export
    DATA_CURATOR = "data_curator" # Dataset submission
    ADMIN = "admin"              # System admin

class User(Base):
    __tablename__ = "users"
    id: int
    email: str
    role: Role = Role.VIEWER  # Default for registered users
    is_active: bool = True

# cytoatlas-api/app/core/permissions.py
from typing import Set

ROLE_PERMISSIONS = {
    Role.ANONYMOUS: {"read:public_data", "search:limited"},
    Role.VIEWER: {"read:all_data", "search:unlimited"},
    Role.RESEARCHER: {"read:all_data", "export:data", "search:unlimited"},
    Role.DATA_CURATOR: {"read:all_data", "export:data", "submit:dataset", "manage:metadata"},
    Role.ADMIN: {"*"},  # All permissions
}

# cytoatlas-api/app/core/security.py
async def check_permission(current_user: User, required_permission: str) -> bool:
    """Verify user has required permission"""
    user_permissions = ROLE_PERMISSIONS[current_user.role]

    # Admin has all permissions
    if "*" in user_permissions:
        return True

    return required_permission in user_permissions

# cytoatlas-api/app/routers/export.py
@router.get("/data/export")
async def export_data(
    current_user: User = Depends(get_current_user),
):
    """Export data as CSV/JSON"""
    if not await check_permission(current_user, "export:data"):
        raise HTTPException(status_code=403, detail="Permission denied")
    # ... export logic
```

## Consequences

### Positive
- **Granular control**: Fine-grained permission management
- **Compliance-ready**: Foundation for audit logging, data governance
- **Scalability**: Easy to add new permissions without changing architecture
- **User satisfaction**: Clear escalation path (anonymous → viewer → researcher → curator)
- **Security baseline**: Explicit permission checks prevent accidental exposure
- **Audit trail**: Who accessed what and when (for compliance)

### Negative
- **Implementation effort**: ~3-5 days (schema, routers, tests, migration)
- **Data migration**: Assign roles to existing users
- **Complexity**: More sophisticated than current `is_admin` boolean
- **Testing burden**: More test cases for permission combinations
- **Breaking change**: Anonymous users lose access to some endpoints

## Alternatives Considered

### Alternative A: Simple Admin/User Binary (Rejected)
- **Pros**: Minimal implementation effort
- **Cons**: Cannot support intermediate use cases (researcher vs curator)
- **Why rejected**: Insufficient granularity for real-world scenarios

### Alternative B: Attribute-Based Access Control (ABAC) (Rejected)
- **Pros**: Ultimate flexibility (user attributes + resource attributes)
- **Cons**: Complex policy engine, harder to audit, over-engineered for current needs
- **Why rejected**: RBAC sufficient for current scale; ABAC available as future upgrade

### Alternative C: Permission-Based (Fine-Grained) (Rejected)
- **Pros**: Extreme flexibility
- **Cons**: Difficult to manage (hundreds of permissions), steep learning curve
- **Why rejected**: Roles provide cleaner abstraction and better UX

## Related ADRs
- [ADR-001: Parquet over JSON](ADR-001-parquet-over-json.md) - Repository pattern can enforce permissions
- [ADR-002: Repository Pattern](ADR-002-repository-pattern.md) - Can add permission checks at repository level

## Implementation Notes

### Phase 1: Database Schema (Round 2, Week 1)

```python
# alembic/versions/001_add_rbac.py
def upgrade():
    # Add role column to users table
    op.add_column('users', sa.Column('role', sa.String(50), nullable=False, server_default='viewer'))

    # Create role enum constraint
    op.execute("""
        CREATE TYPE role_enum AS ENUM ('anonymous', 'viewer', 'researcher', 'data_curator', 'admin');
        ALTER TABLE users ALTER COLUMN role TYPE role_enum;
    """)

def downgrade():
    op.drop_column('users', 'role')
```

### Phase 2: Permission Middleware (Round 2, Week 1-2)

```python
# cytoatlas-api/app/core/security.py
async def check_permission(
    required_permission: str,
    current_user: User = Depends(get_current_user),
):
    """Dependency for permission checking"""
    if not await user_has_permission(current_user, required_permission):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return current_user

# Usage in routers
@router.post("/data/export")
async def export_data(
    _: User = Depends(check_permission("export:data")),
):
    # ... export logic
```

### Phase 3: Audit Logging (Round 2, Week 2-3)

```python
# cytoatlas-api/app/services/audit_service.py
class AuditService:
    async def log_access(
        self,
        user_id: int,
        action: str,
        resource: str,
        ip_address: str,
    ):
        """Log all sensitive data access"""
        audit_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            timestamp=datetime.utcnow(),
        )
        await db.add(audit_entry)
```

### Phase 4: User Migration (Round 2, Week 3)

```sql
-- Assign roles to existing users
UPDATE users SET role = 'admin' WHERE is_admin = TRUE;
UPDATE users SET role = 'viewer' WHERE is_admin = FALSE AND is_active = TRUE;
UPDATE users SET role = 'anonymous' WHERE is_active = FALSE;
```

### Phase 5: Testing (Round 2, Week 2-4)

```python
# tests/test_permissions.py
@pytest.mark.asyncio
async def test_anonymous_cannot_export():
    """Anonymous users cannot export data"""
    response = await client.post(
        "/api/v1/export/data",
        headers={"Authorization": f"Bearer {anonymous_token}"},
    )
    assert response.status_code == 403

@pytest.mark.asyncio
async def test_researcher_can_export():
    """Researcher role can export data"""
    response = await client.post(
        "/api/v1/export/data",
        headers={"Authorization": f"Bearer {researcher_token}"},
    )
    assert response.status_code == 200
```

## Migration Checklist

- [ ] Add `role` column to users table
- [ ] Create role enum
- [ ] Define ROLE_PERMISSIONS mapping
- [ ] Implement `check_permission()` dependency
- [ ] Add permission checks to protected endpoints (~20 endpoints)
- [ ] Implement audit logging
- [ ] Migrate existing users to roles
- [ ] Add tests for all role combinations
- [ ] Document permission model in API docs
- [ ] Deploy to staging, test with real users
- [ ] Deploy to production with grace period (legacy tokens still work)

## Timeline

- **Round 2 Week 1**: Database schema + permissions framework
- **Round 2 Week 2**: Permission middleware + audit logging
- **Round 2 Week 3**: User migration + testing
- **Round 2 Week 4**: Documentation + deployment

## Backward Compatibility

During transition period (grace period):
- Users without assigned roles default to `VIEWER`
- Legacy `is_admin` boolean still respected
- Grace period: 30 days to encourage login with new role system
