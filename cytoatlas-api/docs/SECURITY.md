# Security Architecture

This document describes the security measures implemented in the CytoAtlas API.

## Table of Contents

- [Authentication & Authorization](#authentication--authorization)
- [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
- [Security Headers](#security-headers)
- [Audit Logging](#audit-logging)
- [Input Validation](#input-validation)
- [Dependency Security](#dependency-security)
- [Configuration](#configuration)

## Authentication & Authorization

### JWT Tokens

- HS256 algorithm for token signing
- Configurable expiration (default: 30 minutes)
- Token validation on every request
- User status checks (active, verified)

### API Keys

- Secure generation using `secrets.token_urlsafe(32)`
- Bcrypt hashing for storage
- O(1) lookup via prefix indexing
- Last used timestamp tracking
- Creation timestamp tracking

**API Key Format:**
- Length: 43 characters (base64url encoded)
- Prefix: First 8 characters (indexed for fast lookup)
- Hash: Bcrypt hash stored in database

## Role-Based Access Control (RBAC)

### Roles

1. **ANONYMOUS**: Unauthenticated users
2. **VIEWER**: Authenticated users with basic access
3. **RESEARCHER**: Users who can submit datasets
4. **DATA_CURATOR**: Users who can modify data
5. **ADMIN**: Full system access

### Permissions

- `READ_PUBLIC`: Access public data
- `READ_PRIVATE`: Access private/restricted data
- `WRITE_DATA`: Modify data
- `MANAGE_USERS`: User management operations
- `SUBMIT_DATASET`: Submit datasets for processing
- `USE_CHAT`: Use AI chat assistant

### Role-Permission Mapping

| Role | Permissions |
|------|-------------|
| ANONYMOUS | READ_PUBLIC |
| VIEWER | READ_PUBLIC, USE_CHAT |
| RESEARCHER | READ_PUBLIC, READ_PRIVATE, USE_CHAT, SUBMIT_DATASET |
| DATA_CURATOR | All except MANAGE_USERS |
| ADMIN | All permissions |

### Usage

```python
from fastapi import Depends
from app.core.permissions import require_permission, Permission

@router.get("/private-data")
async def get_private_data(
    _: Annotated[None, Depends(require_permission(Permission.READ_PRIVATE))]
):
    ...
```

## Security Headers

All responses include the following security headers:

- **Content-Security-Policy**: Restricts resource loading
  - Allows self, Plotly, D3.js, and jsDelivr CDN
  - Prevents inline script execution (except for visualization libraries)

- **Strict-Transport-Security** (Production only): Forces HTTPS
  - Max age: 1 year
  - Includes subdomains

- **X-Content-Type-Options**: Prevents MIME sniffing

- **X-Frame-Options**: Prevents clickjacking (DENY)

- **X-XSS-Protection**: Legacy XSS protection

- **Referrer-Policy**: Limits referrer information

- **Permissions-Policy**: Disables dangerous browser features
  - Camera, microphone, geolocation disabled

## Audit Logging

### Features

- Asynchronous, non-blocking writes
- Rotating log files (100 MB, 5 backups)
- Automatic token redaction
- Request ID tracking

### Logged Events

- All write operations (POST, PUT, DELETE, PATCH)
- Authentication attempts (success/failure)
- Rate limit hits
- Error responses (4xx, 5xx)
- User management operations
- Dataset submissions

### Log Format

JSON Lines format with the following fields:

```json
{
  "timestamp": "2026-02-09T12:34:56.789Z",
  "event_type": "auth_success",
  "user_id": 123,
  "ip_address": "192.168.1.1",
  "method": "POST",
  "path": "/api/v1/auth/token",
  "status_code": 200,
  "duration_ms": 45.2,
  "request_id": "uuid-here",
  "details": {}
}
```

### Configuration

```env
AUDIT_ENABLED=true
AUDIT_LOG_PATH=logs/audit.jsonl
```

## Input Validation

### Path Traversal Protection

All file loading operations verify that resolved paths are within allowed directories:

```python
# Blocked patterns:
- ../../etc/passwd
- ../../../secret.json
- /etc/shadow
```

### Request Body Size Limits

- Default: 100 MB
- Configurable via `MAX_REQUEST_BODY_MB`

### CORS Restrictions

- Default allowed origins: `http://localhost:8000,http://localhost:3000`
- Allowed methods: GET, POST, PUT, DELETE, OPTIONS
- Allowed headers: Authorization, Content-Type, X-API-Key, X-Request-ID

## Dependency Security

### Automated Scanning

Run security audit:

```bash
cd cytoatlas-api
./scripts/security_audit.sh
```

This runs:
1. **pip-audit**: Checks for known vulnerabilities in dependencies
2. **bandit**: Scans code for security issues

### CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Security Audit
  run: |
    pip install -e ".[dev]"
    ./scripts/security_audit.sh
```

## Configuration

### Production Requirements

In production (`ENVIRONMENT=production`), the following are **required**:

- `SECRET_KEY`: Must be set to a secure random value
  - Generate: `openssl rand -hex 32`
  - Never use default value
  - Never commit to version control

### Environment Variables

```env
# Required in production
SECRET_KEY=your-secure-random-key-here

# Recommended settings
ENVIRONMENT=production
ALLOWED_ORIGINS=https://cytoatlas.example.com
AUDIT_ENABLED=true
REQUIRE_AUTH=true

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Request limits
MAX_REQUEST_BODY_MB=100
```

### Development Mode

In development, the API will:
- Generate a random SECRET_KEY if not provided (changes on restart)
- Log security warnings for insecure configurations
- Allow CORS from localhost origins
- Expose detailed error messages

### Startup Warnings

The API logs warnings for:
- Missing or default SECRET_KEY in non-production
- CORS wildcard (`*`) configuration
- Disabled audit logging

## Best Practices

### For Administrators

1. **Secret Management**
   - Use environment variables for secrets
   - Rotate SECRET_KEY regularly
   - Use different keys per environment

2. **User Management**
   - Assign minimum required role
   - Regularly audit user roles
   - Disable inactive accounts

3. **Monitoring**
   - Review audit logs regularly
   - Monitor rate limit hits
   - Track authentication failures

### For Developers

1. **Endpoint Security**
   - Use `require_permission()` for protected endpoints
   - Validate all input parameters
   - Never expose internal error details in production

2. **Data Access**
   - Use BaseService.load_json() (includes path traversal protection)
   - Validate file paths before access
   - Sanitize user-provided filenames

3. **Testing**
   - Test permission boundaries
   - Verify RBAC rules
   - Test with different user roles

## Security Considerations

### Known Limitations

1. **SQLite Compatibility**: Database migration doesn't drop `is_admin` column on SQLite (use as computed property instead)

2. **Session Management**: JWT tokens cannot be revoked before expiration (use short expiration times)

3. **Rate Limiting**: In-memory rate limiting doesn't work across multiple workers (use Redis in production)

### Threat Model

Protected against:
- ✅ SQL injection (via SQLAlchemy ORM)
- ✅ Path traversal
- ✅ XSS (via CSP headers)
- ✅ CSRF (via CORS restrictions)
- ✅ Clickjacking (via X-Frame-Options)
- ✅ MIME sniffing
- ✅ Brute force (via rate limiting)

Not protected against:
- ❌ DDoS (requires infrastructure-level protection)
- ❌ Zero-day vulnerabilities in dependencies (requires regular updates)

## Incident Response

If a security issue is discovered:

1. **Immediate Actions**
   - Review audit logs: `tail -f logs/audit.jsonl`
   - Identify affected users
   - Disable compromised accounts

2. **Investigation**
   - Check for unauthorized access patterns
   - Review recent configuration changes
   - Scan for suspicious activities

3. **Remediation**
   - Rotate SECRET_KEY
   - Force password resets if needed
   - Update dependencies
   - Apply security patches

4. **Prevention**
   - Update security policies
   - Add monitoring rules
   - Document lessons learned

## Updates & Maintenance

- Review and update dependencies monthly
- Run security audit before each release
- Monitor security advisories for used packages
- Keep documentation in sync with code

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT Best Practices](https://datatracker.ietf.org/doc/html/rfc8725)
