# CytoAtlas API Manual Security Audit

**Date:** 2026-02-09
**Auditor:** Security Audit Round 3B
**Scope:** Chat service, submit service, auth, configuration, application factory

---

## Finding Summary

| ID | Severity | Component | Finding | Status |
|----|----------|-----------|---------|--------|
| SA-01 | Critical | chat_service.py | No prompt injection defense | FIXED |
| SA-02 | Critical | chat_service.py | System prompt has no security instructions | FIXED |
| SA-03 | High | chat_service.py (legacy) | Error messages leak internal details (str(e)) | FIXED |
| SA-04 | High | chat.py router | Exception handler exposes str(e) to clients | FIXED |
| SA-05 | High | chat.py router | Streaming SSE error leaks str(e) | FIXED |
| SA-06 | High | submit_service.py | No path traversal defense on file_path | FIXED |
| SA-07 | High | submit_service.py | No file extension validation on validate/process | FIXED |
| SA-08 | High | submit_service.py | No file size validation for chunks | FIXED |
| SA-09 | High | submit.py router | validate endpoint leaks internal paths in errors | FIXED |
| SA-10 | Medium | rate_limit.py | IP spoofable via X-Forwarded-For without proxy trust | FIXED |
| SA-11 | Medium | chat.py router | Anonymous rate limit bypassable by changing session_id | FIXED |
| SA-12 | Medium | main.py | Global exception handler leaks str(exc) in debug mode | FIXED |
| SA-13 | Medium | llm_client.py | LLM error messages may contain URLs/paths | FIXED |
| SA-14 | Medium | chat_service.py | No output leakage detection for system prompt | FIXED |
| SA-15 | Medium | chat_service.py | Tool results not validated before LLM consumption | FIXED |
| SA-16 | Low | config.py | No trusted_proxies setting | FIXED |
| SA-17 | Low | security_audit.sh | Audit script lacks secret scanning and detail | FIXED |
| SA-18 | Low | chat_service.py | No input length limit | FIXED |

---

## Detailed Findings

### SA-01: No Prompt Injection Defense (Critical)

**File:** `app/services/chat/chat_service.py`, `app/services/chat_service_legacy.py`

**Description:** User input is passed directly to the LLM without any sanitization or injection detection. An attacker could use prompt injection techniques ("ignore previous instructions", delimiter injection, role manipulation) to override the system prompt, extract internal instructions, or cause the assistant to behave in unintended ways.

**Risk:** An attacker could trick the LLM into revealing the system prompt, bypassing safety guidelines, or executing unintended tool calls.

**Remediation:** Created `app/services/chat/input_sanitizer.py` with:
- `sanitize_user_input()`: Detects 20+ injection patterns across 5 categories (instruction override, role manipulation, system prompt extraction, delimiter injection, code execution)
- Pattern severity levels: critical, high, medium
- Messages flagged as critical/high are blocked before reaching the LLM

### SA-02: System Prompt Missing Security Instructions (Critical)

**File:** `app/services/chat/chat_service.py`

**Description:** The SYSTEM_PROMPT contains no defensive instructions. It does not tell the LLM to refuse system prompt disclosure, resist instruction override attempts, or restrict tool use to intended purposes.

**Remediation:** Added a "Security Instructions" section to SYSTEM_PROMPT with 7 immutable rules covering prompt confidentiality, instruction integrity, tool use restrictions, and data protection.

### SA-03: Legacy Chat Service Leaks Error Details (High)

**File:** `app/services/chat_service_legacy.py`

**Description:** Line 762: `raise RuntimeError(f"Chat service error: {str(e)}")` -- passes internal exception details (which may include API URLs, model names, internal paths) directly to the client.

**Remediation:** Changed to generic error message without internal details.

### SA-04: Chat Router Exposes Internal Errors (High)

**File:** `app/routers/chat.py`

**Description:** Lines 175-179: All exception types including internal errors return `str(e)` to the client via HTTPException. This can expose:
- LLM backend URLs and connection strings
- Internal file paths
- Python traceback information
- API key prefixes in error messages

**Remediation:** Replaced with type-specific handlers: ValueError (safe user messages), PermissionError (generic 403), RuntimeError (generic 503), Exception (generic 500). All log details server-side only.

### SA-05: Streaming SSE Error Leaks Details (High)

**File:** `app/routers/chat.py`

**Description:** The SSE `generate()` function catches all exceptions and returns `str(e)` in the event stream, which is visible to the client.

**Remediation:** Replaced with generic error message; details logged server-side.

### SA-06: No Path Traversal Defense in Submit (High)

**File:** `app/services/submit_service.py`

**Description:** The `validate_h5ad()` and `start_processing()` methods accept a `file_path` string parameter directly from the API without validation. An attacker could pass:
- `../../../etc/passwd` to read arbitrary files
- Symlinks pointing outside the upload directory
- Paths with null bytes to bypass extension checks

**Remediation:** Added `_sanitize_filename()` and `_validate_path_within_directory()` functions:
- Filename sanitization strips directory components, rejects `..`, null bytes
- Path containment check ensures resolved paths are within upload_dir
- Symlink escape detection
- Applied to `init_upload`, `complete_upload`, `validate_h5ad`, and `start_processing`

### SA-07: No File Extension Validation on Validate/Process (High)

**File:** `app/services/submit_service.py`, `app/routers/submit.py`

**Description:** The `validate_h5ad` endpoint accepts any `file_path` from a query parameter without checking file extension or that it is within the expected directory.

**Remediation:** Added extension validation (`.h5ad` only for validate/process, `.h5ad|.csv|.tsv|.json` for upload) and path containment checks.

### SA-08: No Chunk Size Validation (High)

**File:** `app/services/submit_service.py`

**Description:** The `upload_chunk` method accepts arbitrarily large chunks without size validation. An attacker could send oversized chunks to exhaust disk space or memory.

**Remediation:** Added chunk size validation (max 2x declared chunk_size) and cumulative upload size tracking (max 110% of declared file_size).

### SA-09: Submit Router Leaks Internal Paths (High)

**File:** `app/routers/submit.py`

**Description:** Lines 153-154: `raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")` may expose internal file system paths in error messages.

**Remediation:** Split into ValueError (safe to show) and generic catch-all (hide details).

### SA-10: IP Spoofable via X-Forwarded-For (Medium)

**File:** `app/core/rate_limit.py`

**Description:** The rate limiter uses `request.client.host` which can be overridden by X-Forwarded-For header spoofing when not behind a trusted reverse proxy. An attacker can bypass IP-based rate limits by setting arbitrary X-Forwarded-For headers.

**Remediation:** Added `get_real_client_ip()` that only trusts forwarded headers when the direct connection comes from a configured trusted proxy (via `trusted_proxies` setting). Without explicit configuration, forwarded headers are always ignored.

### SA-11: Anonymous Rate Limit Bypass via Session ID (Medium)

**File:** `app/routers/chat.py`

**Description:** Anonymous users are rate-limited by session_id only. An attacker can bypass this by simply not sending (or changing) the session cookie, getting a new session_id for each request.

**Remediation:** Added dual-key rate limiting for anonymous users: both session_id AND IP address are tracked. Exceeding the limit on either key blocks the request.

### SA-12: Global Exception Handler Leaks in Debug Mode (Medium)

**File:** `app/main.py`

**Description:** The global exception handler returns `str(exc)` when `settings.debug` is True. Since `debug` defaults to False, this is low risk, but an accidental `DEBUG=true` in production would expose stack traces and internal paths to all users.

**Remediation:** Changed to only show exception type + sanitized message in development mode, with file paths stripped. Production always returns generic message regardless of debug setting.

### SA-13: LLM Error Messages May Leak Internal Details (Medium)

**File:** `app/services/chat/llm_client.py`

**Description:** LLM client errors (connection failures, API errors) may contain internal URLs, API key fragments, or file paths that could be propagated to the user.

**Remediation:** Added `_sanitize_llm_error()` that strips API keys, URLs, and file paths from error messages before logging. Client-facing errors use generic messages via `raise RuntimeError("LLM service error") from None`.

### SA-14: No System Prompt Leakage Detection (Medium)

**File:** `app/services/chat/chat_service.py`

**Description:** If the LLM inadvertently includes system prompt content in its response (e.g., due to a prompt injection), the verbatim system instructions would be sent to the user.

**Remediation:** Added `check_output_leakage()` that compares LLM responses against significant system prompt fragments and replaces any matches with `[Content removed for security]`.

### SA-15: Tool Results Not Validated (Medium)

**File:** `app/services/chat/chat_service.py`

**Description:** Tool execution results are passed directly to the LLM without validation. Malicious tool results could contain HTML/script injection content that, when rendered in the frontend, could lead to XSS.

**Remediation:** Added `validate_tool_result()` that checks for and strips script injection patterns in tool result values.

### SA-16: No trusted_proxies Setting (Low)

**File:** `app/config.py`

**Description:** No configuration option existed for declaring trusted reverse proxies, which is required for safe X-Forwarded-For handling.

**Remediation:** Added `trusted_proxies: str = ""` setting accepting comma-separated CIDR ranges.

### SA-17: Minimal Security Audit Script (Low)

**File:** `scripts/security_audit.sh`

**Description:** The audit script only ran pip-audit and bandit without secret scanning, configuration checks, or detailed reporting.

**Remediation:** Expanded to 4 phases: dependency scan, static analysis, hardcoded secret scan, and configuration security checks. Added JSON output option, report file generation, and strict mode.

### SA-18: No Input Length Limit (Low)

**File:** `app/services/chat/input_sanitizer.py`

**Description:** User messages had no length limit, allowing an attacker to send extremely large messages that consume LLM tokens/memory.

**Remediation:** Added MAX_INPUT_LENGTH (10,000 characters) with automatic truncation.

---

## Items Verified (No Issues Found)

| Area | Status | Notes |
|------|--------|-------|
| JWT secret_key enforcement | OK | Production validator rejects None/default |
| CORS configuration | OK | Defaults to explicit localhost origins |
| RBAC permissions | OK | 5 roles properly configured |
| API key hashing (PBKDF2) | OK | Prefix index for O(1) lookup |
| Security headers (CSP, HSTS, X-Frame) | OK | Middleware in place |
| Audit logging with token redaction | OK | Functional and tested |
| OAuth2 scheme with optional auth | OK | Endpoints work for both auth/anon |
| File upload requires authentication | OK | All submit endpoints use get_current_user |
| Password hashing (bcrypt) | OK | Proper CryptContext |
| Tool execution sandboxing | OK | Tools limited to read-only CytoAtlas queries |
