# CytoAtlas REST API Reference

Complete reference for all ~262 endpoints across 17 routers. Organized by domain with curl examples.

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
curl -s http://localhost:8000/api/v1/atlases/cima/summary \
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

## 3. CIMA Atlas

Endpoints for CIMA atlas data. Marked `deprecated=True` in favor of unified endpoints.

### Summary

**GET** `/atlases/cima/summary`

Get CIMA atlas summary statistics.

```bash
curl -s http://localhost:8000/api/v1/atlases/cima/summary | jq '.cells, .samples'
```

**Response**: `CIMASummaryStats` | **Auth**: None | **Status**: 200

---

### Available Cell Types

**GET** `/atlases/cima/cell-types`

Get list of available cell types.

```bash
curl -s http://localhost:8000/api/v1/atlases/cima/cell-types | jq '.'

# Response
["CD4 T cells", "CD8 T cells", "B cells", "Monocytes", ...]
```

**Response**: `List[str]` | **Auth**: None | **Status**: 200

---

### Cell Type Activity

**GET** `/atlases/cima/activity`

Get mean activity by cell type.

```bash
curl -s 'http://localhost:8000/api/v1/atlases/cima/activity?signature_type=CytoSig' | jq '.[] | {cell_type, signature, mean_activity}'
```

**Query Parameters**:
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `List[CIMACellTypeActivity]` | **Auth**: None | **Status**: 200

---

### Age Correlations

**GET** `/atlases/cima/correlations/age`

Get activity correlation with age.

```bash
curl -s 'http://localhost:8000/api/v1/atlases/cima/correlations/age?signature_type=CytoSig&offset=0&limit=10' | jq '.'
```

**Query Parameters**:
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")
- `offset`: Pagination offset (default: 0)
- `limit`: Results per page (default: 20, max: 100)

**Response**: `List[CIMACorrelation]` | **Auth**: None | **Status**: 200

---

### BMI Correlations

**GET** `/atlases/cima/correlations/bmi`

Get activity correlation with BMI.

```bash
curl -s 'http://localhost:8000/api/v1/atlases/cima/correlations/bmi?signature_type=CytoSig' | jq '.'
```

**Query Parameters**: Same as age correlations

**Response**: `List[CIMACorrelation]` | **Auth**: None | **Status**: 200

---

## 4. Inflammation Atlas

Endpoints for Inflammation Atlas data. Marked `deprecated=True` in favor of unified endpoints.

### Summary

**GET** `/atlases/inflammation/summary`

Get Inflammation Atlas summary statistics.

```bash
curl -s http://localhost:8000/api/v1/atlases/inflammation/summary | jq '.cells, .samples, .diseases'
```

**Response**: `InflammationSummaryStats` | **Auth**: None | **Status**: 200

---

### Available Diseases

**GET** `/atlases/inflammation/diseases`

Get list of available diseases.

```bash
curl -s http://localhost:8000/api/v1/atlases/inflammation/diseases | jq '.'

# Response
["COVID-19", "Influenza", "Sepsis", "Autoimmune", ...]
```

**Response**: `List[str]` | **Auth**: None | **Status**: 200

---

### Disease Activity

**GET** `/atlases/inflammation/disease-activity`

Get disease-specific activity patterns by cell type.

```bash
curl -s 'http://localhost:8000/api/v1/atlases/inflammation/disease-activity?disease=COVID-19&signature_type=CytoSig' | jq '.'
```

**Query Parameters**:
- `disease`: Disease name (required if filtering)
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `List[InflammationDiseaseActivity]` | **Auth**: None | **Status**: 200

---

### Treatment Response

**GET** `/atlases/inflammation/treatment-response`

Get treatment response prediction results by cell type.

```bash
curl -s 'http://localhost:8000/api/v1/atlases/inflammation/treatment-response?disease=COVID-19' | jq '.'
```

**Query Parameters**:
- `disease`: Filter by disease (optional)

**Response**: `List[InflammationTreatmentResponse]` | **Auth**: None | **Status**: 200

---

## 5. scAtlas

Endpoints for scAtlas data. Marked `deprecated=True` in favor of unified endpoints.

### Summary

**GET** `/atlases/scatlas/summary`

Get scAtlas summary statistics.

```bash
curl -s http://localhost:8000/api/v1/atlases/scatlas/summary | jq '.cells, .organs, .cancer_types'
```

**Response**: `ScAtlasSummaryStats` | **Auth**: None | **Status**: 200

---

### Available Organs

**GET** `/atlases/scatlas/organs`

Get list of available organs/tissues.

```bash
curl -s http://localhost:8000/api/v1/atlases/scatlas/organs | jq '.'

# Response
["Blood", "Bone Marrow", "Lymph Node", "Spleen", "Liver", ...]
```

**Response**: `List[str]` | **Auth**: None | **Status**: 200

---

### Organ Signatures

**GET** `/atlases/scatlas/organ-signatures`

Get cell type signature patterns by organ.

```bash
curl -s 'http://localhost:8000/api/v1/atlases/scatlas/organ-signatures?organ=Blood&signature_type=CytoSig' | jq '.'
```

**Query Parameters**:
- `organ`: Organ name
- `signature_type`: "CytoSig" or "SecAct" (default: "CytoSig")

**Response**: `List[ScAtlasOrganSignature]` | **Auth**: None | **Status**: 200

---

### Cancer Comparison

**GET** `/atlases/scatlas/cancer-comparison`

Compare cancer vs adjacent normal tissue activities.

```bash
curl -s 'http://localhost:8000/api/v1/atlases/scatlas/cancer-comparison?cancer_type=NSCLC&signature_type=CytoSig' | jq '.'
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

## 14. Perturbation (parse_10M + Tahoe)

### Perturbation Summary

**GET** `/perturbation/summary`

Overview statistics for both perturbation datasets.

```bash
curl -s http://localhost:8000/api/v1/perturbation/summary | jq '.'

# Response
{
  "parse10m": {"cells": 9697974, "cytokines": 90, "cell_types": 18, "donors": 12},
  "tahoe": {"cells": 100600000, "drugs": 95, "cell_lines": 50, "plates": 14}
}
```

**Response**: `PerturbationSummary` | **Auth**: None

---

### parse_10M — List Cytokines

**GET** `/perturbation/parse10m/cytokines`

List all 90 cytokine treatment conditions with family annotations.

```bash
curl -s http://localhost:8000/api/v1/perturbation/parse10m/cytokines | jq '.'
```

**Response**: `list[CytokineInfo]` | **Auth**: None

---

### parse_10M — Activity

**GET** `/perturbation/parse10m/activity`

Get activity by cytokine and cell type.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cytokine` | string | No | Filter by cytokine |
| `cell_type` | string | No | Filter by cell type |
| `signature_type` | string | No | CytoSig or SecAct (default: CytoSig) |

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/parse10m/activity?cytokine=IL-17A&cell_type=Th17&signature_type=CytoSig' | jq '.'
```

**Response**: `list[ActivityResult]` | **Auth**: None

---

### parse_10M — Treatment Effect

**GET** `/perturbation/parse10m/treatment-effect`

Treatment vs PBS control differential activity.

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/parse10m/treatment-effect?cell_type=Monocyte&signature_type=CytoSig' | jq '.'
```

**Response**: `list[TreatmentEffect]` | **Auth**: None

---

### parse_10M — Ground Truth Validation

**GET** `/perturbation/parse10m/ground-truth`

CytoSig/SecAct predicted activity vs actual cytokine treatment response.

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/parse10m/ground-truth?signature_type=CytoSig' | jq '.'

# Response
{
  "results": [
    {"cytokine": "IL-17A", "cell_type": "Th17", "self_rank": 1, "auc_roc": 0.95, "self_activity": 2.3},
    {"cytokine": "IFN-gamma", "cell_type": "CD8+ T", "self_rank": 1, "auc_roc": 0.92, "self_activity": 1.8}
  ]
}
```

**Response**: `GroundTruthResults` | **Auth**: None

---

### parse_10M — Heatmap

**GET** `/perturbation/parse10m/heatmap`

Cytokine × cell type heatmap data.

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/parse10m/heatmap?signature_type=CytoSig' | jq '.'
```

**Response**: `HeatmapData` | **Auth**: None

---

### parse_10M — Donor Variability

**GET** `/perturbation/parse10m/donor-variability`

Cross-donor consistency for a given cytokine and cell type.

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/parse10m/donor-variability?cytokine=IL-17A&cell_type=Th17' | jq '.'
```

**Response**: `list[DonorVariability]` | **Auth**: None

---

### parse_10M — Cytokine Families

**GET** `/perturbation/parse10m/cytokine-families`

Cytokines grouped by family (Interleukin, Interferon, TNF superfamily, etc.).

```bash
curl -s http://localhost:8000/api/v1/perturbation/parse10m/cytokine-families | jq '.'
```

**Response**: `list[CytokineFamily]` | **Auth**: None

---

### Tahoe — List Drugs

**GET** `/perturbation/tahoe/drugs`

List all 95 drugs with categories and mechanisms.

```bash
curl -s http://localhost:8000/api/v1/perturbation/tahoe/drugs | jq '.'
```

**Response**: `list[DrugInfo]` | **Auth**: None

---

### Tahoe — List Cell Lines

**GET** `/perturbation/tahoe/cell-lines`

List all 50 cancer cell lines with cancer type annotations.

```bash
curl -s http://localhost:8000/api/v1/perturbation/tahoe/cell-lines | jq '.'
```

**Response**: `list[CellLineInfo]` | **Auth**: None

---

### Tahoe — Drug Effect

**GET** `/perturbation/tahoe/drug-effect`

Drug vs DMSO differential activity.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `drug` | string | No | Filter by drug name |
| `cell_line` | string | No | Filter by cell line |
| `signature_type` | string | No | CytoSig or SecAct (default: CytoSig) |

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/tahoe/drug-effect?drug=Trametinib&cell_line=A549&signature_type=CytoSig' | jq '.'
```

**Response**: `list[DrugEffect]` | **Auth**: None

---

### Tahoe — Sensitivity Matrix

**GET** `/perturbation/tahoe/sensitivity-matrix`

Drug × cell line sensitivity matrix.

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/tahoe/sensitivity-matrix?signature_type=CytoSig' | jq '.'
```

**Response**: `SensitivityMatrix` | **Auth**: None

---

### Tahoe — Dose-Response

**GET** `/perturbation/tahoe/dose-response`

Plate 13 dose-response data (3 dose levels × 25 drugs × 50 cell lines).

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/tahoe/dose-response?drug=Trametinib&cell_line=A549' | jq '.'

# Response
{
  "drug": "Trametinib",
  "cell_line": "A549",
  "doses": [0.1, 1.0, 10.0],
  "signatures": [
    {"name": "IL6", "activities": [-0.5, -1.2, -2.1]},
    {"name": "TNF", "activities": [-0.3, -0.8, -1.5]}
  ]
}
```

**Response**: `DoseResponse` | **Auth**: None

---

### Tahoe — Pathway Activation

**GET** `/perturbation/tahoe/pathway-activation`

Drug → cytokine pathway mapping.

```bash
curl -s 'http://localhost:8000/api/v1/perturbation/tahoe/pathway-activation?drug=Bortezomib' | jq '.'
```

**Response**: `list[PathwayActivation]` | **Auth**: None

---

## 15. Spatial (SpatialCorpus-110M)

### Spatial Summary

**GET** `/spatial/summary`

Overview: 251 datasets, 8 technologies, tissue distribution.

```bash
curl -s http://localhost:8000/api/v1/spatial/summary | jq '.'

# Response
{
  "total_datasets": 251,
  "human_datasets": 244,
  "technologies": 8,
  "total_cells": 110000000,
  "tier_a_files": 171,
  "tier_b_files": 51,
  "tier_c_files": 12
}
```

**Response**: `SpatialSummary` | **Auth**: None

---

### Spatial — Technologies

**GET** `/spatial/technologies`

List technologies with dataset counts and gene panel sizes.

```bash
curl -s http://localhost:8000/api/v1/spatial/technologies | jq '.'
```

**Response**: `list[TechnologyInfo]` | **Auth**: None

---

### Spatial — Tissues

**GET** `/spatial/tissues`

List tissues with dataset counts.

```bash
curl -s http://localhost:8000/api/v1/spatial/tissues | jq '.'
```

**Response**: `list[TissueInfo]` | **Auth**: None

---

### Spatial — Dataset Catalog

**GET** `/spatial/datasets`

Full dataset catalog with metadata.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `technology` | string | No | Filter by technology |
| `tissue` | string | No | Filter by tissue |

```bash
curl -s 'http://localhost:8000/api/v1/spatial/datasets?technology=Visium&tissue=Brain' | jq '.'
```

**Response**: `list[SpatialDataset]` | **Auth**: None

---

### Spatial — Activity

**GET** `/spatial/activity`

Activity by technology and tissue.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `technology` | string | No | Filter by technology |
| `tissue` | string | No | Filter by tissue |
| `signature_type` | string | No | CytoSig or SecAct (default: CytoSig) |

```bash
curl -s 'http://localhost:8000/api/v1/spatial/activity?technology=Visium&tissue=Brain&signature_type=CytoSig' | jq '.'
```

**Response**: `list[SpatialActivity]` | **Auth**: None

---

### Spatial — Tissue Summary

**GET** `/spatial/tissue-summary`

Tissue-level activity summary across technologies.

```bash
curl -s 'http://localhost:8000/api/v1/spatial/tissue-summary?signature_type=CytoSig' | jq '.'
```

**Response**: `list[TissueSummary]` | **Auth**: None

---

### Spatial — Neighborhood Activity

**GET** `/spatial/neighborhood`

Niche-level activity patterns.

```bash
curl -s 'http://localhost:8000/api/v1/spatial/neighborhood?tissue=Breast&signature_type=CytoSig' | jq '.'
```

**Response**: `list[NeighborhoodActivity]` | **Auth**: None

---

### Spatial — Technology Comparison

**GET** `/spatial/technology-comparison`

Cross-technology reproducibility for same tissues.

```bash
curl -s 'http://localhost:8000/api/v1/spatial/technology-comparison?signature_type=CytoSig' | jq '.'
```

**Response**: `TechnologyComparison` | **Auth**: None

---

### Spatial — Gene Coverage

**GET** `/spatial/gene-coverage`

Gene panel coverage vs CytoSig/SecAct per technology.

```bash
curl -s 'http://localhost:8000/api/v1/spatial/gene-coverage?technology=Xenium' | jq '.'

# Response
{
  "technology": "Xenium",
  "total_genes": 400,
  "cytosig_overlap": 18,
  "cytosig_coverage": 0.12,
  "secact_overlap": 45,
  "secact_coverage": 0.036
}
```

**Response**: `GeneCoverage` | **Auth**: None

---

### Spatial — Coordinates

**GET** `/spatial/coordinates/{dataset_id}`

Sampled spatial coordinates for visualization (downsampled for API).

```bash
curl -s http://localhost:8000/api/v1/spatial/coordinates/visium_brain_001 | jq '.'
```

**Response**: `SpatialCoordinates` | **Auth**: None

---

## 16. OpenAPI Documentation

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
curl -s 'http://localhost:8000/api/v1/atlases/cima/activity?signature_type=CytoSig'

# Get disease activity for COVID-19
curl -s 'http://localhost:8000/api/v1/atlases/inflammation/disease-activity?disease=COVID-19'

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
curl -s http://localhost:8000/api/v1/atlases/cima/summary \
  -H "X-API-Key: sk-1234567890abcdef"
```

---

For more details, see [docs/DEPLOYMENT.md](DEPLOYMENT.md) for deployment instructions or [docs/USER_GUIDE.md](USER_GUIDE.md) for usage examples.
