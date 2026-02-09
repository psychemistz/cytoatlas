# CytoAtlas REST API Reference

Complete reference for all 188+ endpoints across 14 routers. Organized by domain with curl examples.

**Last Updated**: 2026-02-09
**Base URL**: `/api/v1`
**Authentication**: Optional (defaults to anonymous role)

---

## 1. Health & Status

### Health Check

**GET** `/health`

Check API health status (database, cache, environment).

```bash
curl -s http://localhost:8000/api/v1/health | jq '.'

# Response
{
  "status": "healthy",
  "version": "0.1.0",
  "database": "not configured",
  "cache": "in-memory",
  "environment": "development"
}
```

**Response**: `HealthResponse` | **Auth**: None

---

### Readiness Probe

**GET** `/health/ready`

Kubernetes readiness probe (200 if service ready).

```bash
curl -s http://localhost:8000/api/v1/health/ready | jq '.'

# Response (if ready)
{ "ready": true }

# Response (if not ready)
{ "ready": false }
```

**Response**: `{ ready: bool }` | **Auth**: None

---

### Liveness Probe

**GET** `/health/live`

Kubernetes liveness probe (200 if service alive).

```bash
curl -s http://localhost:8000/api/v1/health/live | jq '.'

# Response
{ "alive": true }
```

**Response**: `{ alive: bool }` | **Auth**: None

---

### Metrics

**GET** `/health/metrics`

Get API metrics summary (request counts, response times, error rates).

```bash
curl -s http://localhost:8000/api/v1/health/metrics | jq '.'

# Response
{
  "total_requests": 1542,
  "total_errors": 3,
  "cache_hits": 1200,
  "cache_misses": 342,
  "avg_response_time_ms": 145
}
```

**Response**: `{ total_requests: int, total_errors: int, ... }` | **Auth**: None

---

## 2. Authentication

### Login / Get Token

**POST** `/auth/token`

Authenticate user with email and password to receive JWT token.

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=yourpassword"

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Response**: `TokenResponse` | **Auth**: None | **Status**: 200, 401

---

### Register User

**POST** `/auth/register`

Register a new user account.

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@example.com",
    "password": "secure_password",
    "full_name": "John Doe",
    "institution": "NIH"
  }'

# Response
{
  "id": 1,
  "email": "newuser@example.com",
  "full_name": "John Doe",
  "institution": "NIH",
  "is_active": true,
  "is_admin": false
}
```

**Response**: `UserResponse` | **Auth**: None | **Status**: 201, 400

---

### Get Current User

**GET** `/auth/me`

Get current authenticated user information.

```bash
curl -s http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" | jq '.'
```

**Response**: `UserResponse` | **Auth**: Required | **Status**: 200, 401

---

### Generate API Key

**POST** `/auth/api-key`

Generate a new API key for current user (replace existing).

```bash
curl -X POST http://localhost:8000/api/v1/auth/api-key \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Response (shown only once)
{
  "api_key": "sk-1234567890abcdef",
  "message": "Store this key securely. It will not be shown again."
}

# Use in subsequent requests
curl -s http://localhost:8000/api/v1/cima/summary \
  -H "X-API-Key: sk-1234567890abcdef"
```

**Response**: `APIKeyResponse` | **Auth**: Required | **Status**: 200

---

### Revoke API Key

**DELETE** `/auth/api-key`

Revoke current API key.

```bash
curl -X DELETE http://localhost:8000/api/v1/auth/api-key \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Response
{ "message": "API key revoked successfully" }
```

**Response**: `{ message: str }` | **Auth**: Required | **Status**: 200

---

## 3. CIMA Atlas (Legacy - Deprecated)

Endpoints for CIMA atlas data. Marked `deprecated=True` in favor of unified endpoints.

### Summary

**GET** `/cima/summary`

Get CIMA atlas summary statistics.

```bash
curl -s http://localhost:8000/api/v1/cima/summary | jq '.cells, .samples'
```

**Response**: `CIMASummaryStats` | **Auth**: None | **Status**: 200

---

### Available Cell Types

**GET** `/cima/cell-types`

Get list of available cell types.

```bash
curl -s http://localhost:8000/api/v1/cima/cell-types | jq '.'

# Response
["CD4 T cells", "CD8 T cells", "B cells", "Monocytes", ...]
```

**Response**: `List[str]` | **Auth**: None | **Status**: 200

---

### Cell Type Activity

**GET** `/cima/activity`

Get mean activity by cell type.

```bash
curl -s 'http://localhost:8000/api/v1/cima/activity?signature_type=CytoSig' | jq '.[] | {cell_type, signature, mean_activity}'
```

**Query Parameters**:
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `List[CIMACellTypeActivity]` | **Auth**: None | **Status**: 200

---

### Age Correlations

**GET** `/cima/correlations/age`

Get activity correlation with age.

```bash
curl -s 'http://localhost:8000/api/v1/cima/correlations/age?signature_type=CytoSig&offset=0&limit=10' | jq '.'
```

**Query Parameters**:
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")
- `offset`: Pagination offset (default: 0)
- `limit`: Results per page (default: 20, max: 100)

**Response**: `List[CIMACorrelation]` | **Auth**: None | **Status**: 200

---

### BMI Correlations

**GET** `/cima/correlations/bmi`

Get activity correlation with BMI.

```bash
curl -s 'http://localhost:8000/api/v1/cima/correlations/bmi?signature_type=CytoSig' | jq '.'
```

**Query Parameters**: Same as age correlations

**Response**: `List[CIMACorrelation]` | **Auth**: None | **Status**: 200

---

## 4. Inflammation Atlas (Legacy - Deprecated)

Endpoints for Inflammation Atlas data. Marked `deprecated=True` in favor of unified endpoints.

### Summary

**GET** `/inflammation/summary`

Get Inflammation Atlas summary statistics.

```bash
curl -s http://localhost:8000/api/v1/inflammation/summary | jq '.cells, .samples, .diseases'
```

**Response**: `InflammationSummaryStats` | **Auth**: None | **Status**: 200

---

### Available Diseases

**GET** `/inflammation/diseases`

Get list of available diseases.

```bash
curl -s http://localhost:8000/api/v1/inflammation/diseases | jq '.'

# Response
["COVID-19", "Influenza", "Sepsis", "Autoimmune", ...]
```

**Response**: `List[str]` | **Auth**: None | **Status**: 200

---

### Disease Activity

**GET** `/inflammation/disease-activity`

Get disease-specific activity patterns by cell type.

```bash
curl -s 'http://localhost:8000/api/v1/inflammation/disease-activity?disease=COVID-19&signature_type=CytoSig' | jq '.'
```

**Query Parameters**:
- `disease`: Disease name (required if filtering)
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `List[InflammationDiseaseActivity]` | **Auth**: None | **Status**: 200

---

### Treatment Response

**GET** `/inflammation/treatment-response`

Get treatment response prediction results by cell type.

```bash
curl -s 'http://localhost:8000/api/v1/inflammation/treatment-response?disease=COVID-19' | jq '.'
```

**Query Parameters**:
- `disease`: Filter by disease (optional)

**Response**: `List[InflammationTreatmentResponse]` | **Auth**: None | **Status**: 200

---

## 5. scAtlas (Legacy - Deprecated)

Endpoints for scAtlas data. Marked `deprecated=True` in favor of unified endpoints.

### Summary

**GET** `/scatlas/summary`

Get scAtlas summary statistics.

```bash
curl -s http://localhost:8000/api/v1/scatlas/summary | jq '.cells, .organs, .cancer_types'
```

**Response**: `ScAtlasSummaryStats` | **Auth**: None | **Status**: 200

---

### Available Organs

**GET** `/scatlas/organs`

Get list of available organs/tissues.

```bash
curl -s http://localhost:8000/api/v1/scatlas/organs | jq '.'

# Response
["Blood", "Bone Marrow", "Lymph Node", "Spleen", "Liver", ...]
```

**Response**: `List[str]` | **Auth**: None | **Status**: 200

---

### Organ Signatures

**GET** `/scatlas/organ-signatures`

Get cell type signature patterns by organ.

```bash
curl -s 'http://localhost:8000/api/v1/scatlas/organ-signatures?organ=Blood&signature_type=CytoSig' | jq '.'
```

**Query Parameters**:
- `organ`: Organ name
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `List[ScAtlasOrganSignature]` | **Auth**: None | **Status**: 200

---

### Cancer Comparison

**GET** `/scatlas/cancer-comparison`

Compare cancer vs adjacent normal tissue activities.

```bash
curl -s 'http://localhost:8000/api/v1/scatlas/cancer-comparison?cancer_type=NSCLC&signature_type=CytoSig' | jq '.'
```

**Query Parameters**:
- `cancer_type`: Cancer type (optional)
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `List[ScAtlasCancerComparison]` | **Auth**: None | **Status**: 200

---

## 6. Cross-Atlas Comparison

Compare signatures and cell types across atlases.

### Available Atlases

**GET** `/cross-atlas/atlases`

Get list of available atlases for comparison.

```bash
curl -s http://localhost:8000/api/v1/cross-atlas/atlases | jq '.'

# Response
["CIMA", "Inflammation", "scAtlas"]
```

**Response**: `List[str]` | **Auth**: None | **Status**: 200

---

### Cross-Atlas Summary

**GET** `/cross-atlas/summary`

Get cross-atlas summary statistics.

```bash
curl -s http://localhost:8000/api/v1/cross-atlas/summary | jq '.'
```

**Response**: `dict` | **Auth**: None | **Status**: 200

---

### Cell Type Sankey

**GET** `/cross-atlas/celltype-sankey`

Get cell type mapping data for Sankey visualization.

```bash
curl -s 'http://localhost:8000/api/v1/cross-atlas/celltype-sankey?level=coarse&lineage=all' | jq '.'
```

**Query Parameters**:
- `level`: "coarse" (8 lineages) or "fine" (~32 types) (default: coarse)
- `lineage`: "all", "T_cell", "Myeloid", "B_cell", "NK_ILC" (default: all)

**Response**: `dict` | **Auth**: None | **Status**: 200

---

### Pairwise Scatter

**GET** `/cross-atlas/pairwise-scatter`

Get scatter plot data for atlas comparison.

```bash
curl -s 'http://localhost:8000/api/v1/cross-atlas/pairwise-scatter?atlas1=CIMA&atlas2=Inflammation&signature_type=CytoSig&level=coarse&view=pseudobulk' | jq '.'
```

**Query Parameters**:
- `atlas1`: First atlas (default: "CIMA")
- `atlas2`: Second atlas (default: "Inflammation")
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")
- `level`: "coarse" or "fine" (default: coarse)
- `view`: "pseudobulk" or "singlecell" (default: pseudobulk)

**Response**: `dict` | **Auth**: None | **Status**: 200

---

## 7. Validation & Credibility

Five types of validation for assessing CytoSig/SecAct inference credibility.

### Validation Summary

**GET** `/validation/summary/{atlas}`

Get overall validation summary with quality grade.

```bash
curl -s 'http://localhost:8000/api/v1/validation/summary/CIMA?signature_type=CytoSig' | jq '.'

# Response includes
{
  "atlas": "CIMA",
  "signature_type": "CytoSig",
  "quality_score": 87,
  "quality_grade": "A",
  "components": {
    "expression_correlation": 0.92,
    "gene_coverage": 0.88,
    "stability": 0.85,
    "biological_concordance": 0.84
  }
}
```

**Query Parameters**:
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `ValidationSummary` | **Auth**: None | **Status**: 200

---

### Type 1: Sample-Level Validation

**GET** `/validation/sample-level/{atlas}/{signature}`

Sample-level: Pseudobulk expression vs sample-level activity.

```bash
curl -s 'http://localhost:8000/api/v1/validation/sample-level/CIMA/IL17A' | jq '.'
```

**Response**: `SampleLevelValidation` | **Auth**: None | **Status**: 200

---

### Type 2: Cell Type-Level Validation

**GET** `/validation/celltype-level/{atlas}/{signature}`

Cell type-level: Cell type pseudobulk expression vs activity.

```bash
curl -s 'http://localhost:8000/api/v1/validation/celltype-level/CIMA/IL17A' | jq '.'
```

**Response**: `CellTypeLevelValidation` | **Auth**: None | **Status**: 200

---

### Type 3: Pseudobulk vs Single-Cell

**GET** `/validation/pseudobulk-vs-singlecell/{atlas}/{signature}`

Compare aggregation methods (pseudobulk vs direct single-cell).

```bash
curl -s 'http://localhost:8000/api/v1/validation/pseudobulk-vs-singlecell/CIMA/IL17A' | jq '.'
```

**Response**: `PseudobulkVsSingleCellValidation` | **Auth**: None | **Status**: 200

---

### Type 4: Single-Cell Direct

**GET** `/validation/singlecell-direct/{atlas}/{signature}`

Expression vs activity at cell level (single-cell direct).

```bash
curl -s 'http://localhost:8000/api/v1/validation/singlecell-direct/CIMA/IL17A' | jq '.'
```

**Response**: `SingleCellDirectValidation` | **Auth**: None | **Status**: 200

---

### Type 5: Biological Associations

**GET** `/validation/biological-associations/{atlas}`

Known marker validation (biological concordance).

```bash
curl -s 'http://localhost:8000/api/v1/validation/biological-associations/CIMA?signature_type=CytoSig' | jq '.'
```

**Response**: `BiologicalValidationTable` | **Auth**: None | **Status**: 200

---

## 8. Search

Global search for genes, cytokines, proteins, cell types, diseases, organs.

### Full-Text Search

**GET** `/search`

Search across all indexed entities with fuzzy matching.

```bash
curl -s 'http://localhost:8000/api/v1/search?q=IFNG&type=all&offset=0&limit=20' | jq '.'

# Response
{
  "query": "IFNG",
  "total": 42,
  "results": [
    {
      "entity_type": "cytokine",
      "name": "IFNG",
      "display_name": "Interferon gamma",
      "description": "Pro-inflammatory cytokine",
      "relevance_score": 0.99
    },
    {
      "entity_type": "gene",
      "name": "IFNG",
      "display_name": "IFNG (Gene)",
      "relevance_score": 0.95
    }
  ]
}
```

**Query Parameters**:
- `q`: Search query (required, min 1 char)
- `type`: "gene", "cytokine", "protein", "cell_type", "disease", "organ", "all" (default: all)
- `offset`: Pagination offset (default: 0)
- `limit`: Results per page (default: 20, max: 100)

**Response**: `SearchResponse` | **Auth**: None | **Status**: 200

---

### Autocomplete

**GET** `/search/autocomplete`

Get autocomplete suggestions for search queries.

```bash
curl -s 'http://localhost:8000/api/v1/search/autocomplete?q=IF&limit=10' | jq '.'

# Response
{
  "suggestions": [
    {
      "text": "IFNG",
      "highlighted": "IFNG (Interferon gamma)",
      "entity_type": "cytokine"
    },
    {
      "text": "IFN-alpha",
      "highlighted": "IFN-alpha (Type I interferon)",
      "entity_type": "cytokine"
    }
  ]
}
```

**Query Parameters**:
- `q`: Partial query for suggestions (required, min 1 char)
- `limit`: Maximum suggestions (default: 10, max: 20)

**Response**: `AutocompleteResponse` | **Auth**: None | **Status**: 200

---

## 9. Chat

LLM-powered natural language interface with RAG.

### Send Message

**POST** `/chat/conversations`

Start new conversation or send message to existing conversation.

```bash
curl -X POST http://localhost:8000/api/v1/chat/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What cytokines are elevated in COVID-19?",
    "conversation_id": null
  }'

# Response
{
  "conversation_id": "conv_abc123",
  "role": "assistant",
  "content": "Based on the CytoAtlas data, COVID-19 shows elevated levels of TNF-alpha, IL-6, and IL-8...",
  "sources": [
    {
      "file": "inflammation_disease_activity.json",
      "disease": "COVID-19",
      "cell_types": ["Monocytes", "Macrophages"]
    }
  ]
}
```

**Request Body**:
- `message`: User message (required)
- `conversation_id`: Existing conversation ID (optional, null for new)

**Response**: `ChatMessageResponse` | **Auth**: Optional | **Status**: 200, 429

---

### Stream Response

**GET** `/chat/conversations/{conversation_id}/stream`

Stream chat response (Server-Sent Events).

```bash
curl -N http://localhost:8000/api/v1/chat/conversations/conv_abc123/stream
```

**Response**: SSE stream of `ChatMessageResponse` | **Auth**: Optional | **Status**: 200

---

### Get Conversation History

**GET** `/chat/conversations/{conversation_id}`

Get conversation history.

```bash
curl -s http://localhost:8000/api/v1/chat/conversations/conv_abc123 | jq '.'

# Response
{
  "conversation_id": "conv_abc123",
  "created_at": "2026-02-09T10:30:45Z",
  "messages": [
    {
      "role": "user",
      "content": "What cytokines are elevated in COVID-19?"
    },
    {
      "role": "assistant",
      "content": "Based on the CytoAtlas data..."
    }
  ]
}
```

**Response**: `ConversationResponse` | **Auth**: Optional | **Status**: 200, 404

---

### Get Chat Suggestions

**GET** `/chat/suggestions`

Get suggested chat topics.

```bash
curl -s http://localhost:8000/api/v1/chat/suggestions | jq '.'

# Response
{
  "suggestions": [
    {
      "title": "Disease Activity Patterns",
      "description": "Explore cytokine activity in different diseases",
      "prompt": "Show me the top cytokines elevated in COVID-19"
    },
    {
      "title": "Cross-Atlas Comparison",
      "description": "Compare signatures across atlases",
      "prompt": "Which cytokines are conserved across all three atlases?"
    }
  ]
}
```

**Response**: `ChatSuggestionsResponse` | **Auth**: None | **Status**: 200

---

## 10. Data Export

Export data as CSV or JSON.

### Export CIMA Correlations

**GET** `/export/cima/correlations`

Export CIMA correlations as CSV.

```bash
curl -s 'http://localhost:8000/api/v1/export/cima/correlations?signature_type=CytoSig' \
  -H "Accept: text/csv" \
  -o correlations.csv
```

**Query Parameters**:
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: CSV file | **Auth**: Optional | **Status**: 200

---

### Export Disease Activity

**GET** `/export/inflammation/disease-activity`

Export disease activity as CSV or JSON.

```bash
curl -s 'http://localhost:8000/api/v1/export/inflammation/disease-activity?disease=COVID-19' \
  -H "Accept: text/csv" \
  -o disease_activity.csv
```

**Query Parameters**:
- `disease`: Filter by disease (optional)

**Response**: CSV or JSON file | **Auth**: Optional | **Status**: 200

---

## 11. Pipeline Management

Manage analysis pipeline execution and monitoring.

### Get Pipeline Status

**GET** `/pipeline/status`

Get current pipeline execution status.

```bash
curl -s http://localhost:8000/api/v1/pipeline/status | jq '.'

# Response
{
  "status": "running",
  "current_step": "02_inflam_activity.py",
  "progress": 45,
  "start_time": "2026-02-09T08:00:00Z",
  "estimated_completion": "2026-02-09T14:30:00Z"
}
```

**Response**: `dict` | **Auth**: None | **Status**: 200

---

## 12. Rate Limiting

API enforces rate limits:

- **Anonymous users**: 100 requests/minute, 5 chat messages/day
- **Authenticated users**: 1000 requests/minute, 1000 chat messages/day

Rate limit headers in response:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1644408645
```

Exceeding limit returns `429 Too Many Requests`.

---

## 13. Error Responses

All errors follow RFC 7807 (Problem Details for HTTP APIs):

```json
{
  "detail": "Not Found",
  "status": 404,
  "title": "Item not found",
  "type": "http://example.com/errors/not-found"
}
```

Common status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

---

## 14. OpenAPI Documentation

Full interactive API documentation available at:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI JSON**: `/openapi.json`

Example:

```bash
# View Swagger UI
open http://localhost:8000/docs

# View ReDoc
open http://localhost:8000/redoc

# Download OpenAPI spec
curl -s http://localhost:8000/openapi.json > openapi.json
```

---

## Quick Reference

### Common Queries

```bash
# Get all cytokine activities in CIMA
curl -s 'http://localhost:8000/api/v1/cima/activity?signature_type=CytoSig'

# Get disease activity for COVID-19
curl -s 'http://localhost:8000/api/v1/inflammation/disease-activity?disease=COVID-19'

# Compare CIMA vs Inflammation atlases
curl -s 'http://localhost:8000/api/v1/cross-atlas/pairwise-scatter?atlas1=CIMA&atlas2=Inflammation'

# Get validation grade for CIMA
curl -s 'http://localhost:8000/api/v1/validation/summary/CIMA'

# Search for interferon-related
curl -s 'http://localhost:8000/api/v1/search?q=interferon'
```

### Authentication Examples

```bash
# Using JWT token
curl -s http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Using API key
curl -s http://localhost:8000/api/v1/cima/summary \
  -H "X-API-Key: sk-1234567890abcdef"
```

---

For more details, see [docs/DEPLOYMENT.md](DEPLOYMENT.md) for deployment instructions or [docs/USER_GUIDE.md](USER_GUIDE.md) for usage examples.
