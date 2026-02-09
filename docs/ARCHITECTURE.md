# CytoAtlas System Architecture

Comprehensive architecture documentation for the Pan-Disease Single-Cell Cytokine Activity Atlas project.

**Last Updated**: 2026-02-09
**Scope**: End-to-end system from raw data to REST API to web visualization

---

## 1. System Overview

### 1.1 Purpose

The CytoAtlas project computes **cytokine and secreted protein activity signatures** across 12+ million human immune cells from three major single-cell atlases (CIMA, Inflammation Atlas, scAtlas) to identify disease-specific and conserved signaling patterns. Activity signatures are made available via:
- REST API (188+ endpoints, FastAPI)
- Web dashboard (Single-Page Application)
- Interactive panels (Plotly, D3.js visualization)

### 1.2 Key Metrics

| Metric | Value |
|--------|-------|
| **Total cells analyzed** | 17M+ |
| **Single-cell atlases** | 3 (CIMA, Inflammation, scAtlas) |
| **Signature types** | 3 (CytoSig 44, LinCytoSig 178, SecAct 1,249) |
| **REST API endpoints** | 188+ |
| **API routers** | 14 |
| **Analysis scripts** | 7 Python pipelines + 5 SLURM batches |
| **Web UI pages** | 8 (Landing, Explore, Compare, Validate, Submit, Chat, etc.) |
| **JSON visualization files** | 30+ (~500MB) |

### 1.3 High-Level Data Flow

```
Raw H5AD Files (282GB)
   ↓
Analysis Pipelines (scripts/00-07_*.py)
   ├─ GPU acceleration (CuPy)
   ├─ Pseudo-bulk aggregation
   └─ Ridge regression activity inference
   ↓
Results Directory (CSV, H5AD)
   ↓
Preprocessing (06_preprocess_viz_data.py)
   ├─ Filter/transform results
   └─ Generate JSON files
   ↓
visualization/data/ (30+ JSON files)
   ↓
API Layer (FastAPI + Services)
   ├─ Load JSON with caching
   ├─ Filter/transform per request
   └─ Return JSON responses
   ↓
Frontend SPA (JavaScript)
   └─ Interactive visualization (Plotly, D3.js)
```

---

## 2. Component Inventory

### 2.1 Data Processing Pipeline

| Component | Location | Technology | Purpose |
|-----------|----------|-----------|---------|
| **Raw Data** | `/data/Jiang_Lab/Data/Seongyong/` | H5AD files (AnnData) | Source: 3 single-cell atlases |
| **Pilot Analysis** | `scripts/00_pilot_analysis.py` | Python/NumPy | Validate 100K cell subsets |
| **CIMA Analysis** | `scripts/01_cima_activity.py` | Python/CuPy | 6.5M cells → correlations + biochemistry |
| **Inflammation Analysis** | `scripts/02_inflam_activity.py` | Python/CuPy | 6.3M cells → disease activity + treatment response |
| **scAtlas Analysis** | `scripts/03_scatlas_analysis.py` | Python/CuPy | 6.4M cells → organ signatures + cancer comparison |
| **Integrated Analysis** | `scripts/04_integrated.py` | Python/SciPy | Cross-atlas comparison |
| **Figures** | `scripts/05_figures.py` | Matplotlib/Seaborn | Publication-quality visualizations |
| **Preprocessing** | `scripts/06_preprocess_viz_data.py` | Python/Pandas | H5AD + CSV → JSON for web |
| **Immune Analysis** | `scripts/07_scatlas_immune_analysis.py` | Python/CuPy | T cell exhaustion + infiltration |

### 2.2 REST API Backend

| Component | Location | Files | Purpose |
|-----------|----------|-------|---------|
| **FastAPI App** | `cytoatlas-api/app/main.py` | 1 | Application factory with lifespan |
| **Configuration** | `cytoatlas-api/app/config.py` | 1 | Pydantic settings, environment variables |
| **Routers** | `cytoatlas-api/app/routers/` | 14 files | 188+ endpoints across 14 categories |
| **Services** | `cytoatlas-api/app/services/` | 12 files | Business logic, JSON loading, caching |
| **Schemas** | `cytoatlas-api/app/schemas/` | 10 files | Pydantic v2 request/response models |
| **Core** | `cytoatlas-api/app/core/` | 5 files | Security, cache, database, logging |
| **Models** | `cytoatlas-api/app/models/` | 9 files | SQLAlchemy ORM (future DB integration) |

### 2.3 Web Portal (SPA)

| Component | Location | Files | Purpose |
|-----------|----------|-------|---------|
| **Entry Point** | `cytoatlas-api/static/index.html` | 1 | SPA shell |
| **Main App** | `cytoatlas-api/static/js/app.js` | 1 | Application orchestration |
| **Router** | `cytoatlas-api/static/js/router.js` | 1 | Client-side routing |
| **API Client** | `cytoatlas-api/static/js/api.js` | 1 | Fetch wrapper + caching |
| **Pages** | `cytoatlas-api/static/js/pages/` | 8 | Landing, Explore, Compare, Validate, Chat, etc. |
| **Components** | `cytoatlas-api/static/js/components/` | 20+ | Reusable UI components |
| **Styling** | `cytoatlas-api/static/css/` | Multiple | Responsive design, theme |

### 2.4 Validation System

| Component | Location | Type | Coverage |
|-----------|----------|------|----------|
| **5-Type Validation** | `cytoatlas-api/app/services/validation_service.py` | Python service | 636 lines, all 3 atlases |
| **Validation Data** | `visualization/data/validation/*.json` | JSON | ~175-336MB per atlas |
| **Validation UI** | `cytoatlas-api/static/js/pages/validate.js` | JavaScript | 4 tabs (sample, celltype, SC, summary) |
| **Endpoints** | `cytoatlas-api/app/routers/validation.py` | FastAPI | 12 endpoints for quality metrics |

---

## 3. Data Flow Architecture

### 3.1 From Raw H5AD to JSON

```
Step 1: Raw Data
├─ CIMA_RNA_6484974cells_36326genes_compressed.h5ad (120GB)
├─ INFLAMMATION_ATLAS_main_afterQC.h5ad (100GB)
└─ igt_s9_fine_counts.h5ad (62GB)

Step 2: Analysis Scripts (GPU acceleration)
├─ Load H5AD with backed='r' (memory efficient)
├─ Extract cell × gene expression matrix
├─ Aggregate by pseudo-bulk (sample × cell type)
├─ Run ridge regression: signature_matrix → activities
├─ Compute statistics (mean, std, p-value)
└─ Save to CSV + H5AD results

Step 3: Results Organization
├─ results/cima/*.csv (correlations, differential, etc.)
├─ results/inflammation/*.csv
├─ results/scatlas/*.csv
└─ results/integrated/*.csv

Step 4: Preprocessing (06_preprocess_viz_data.py)
├─ Load results CSVs
├─ Filter (e.g., FDR < 0.05, r > 0.5)
├─ Transform (z-scores, log-fold-change)
├─ Aggregate (top N, grouping)
└─ Convert to JSON

Step 5: Web Data
├─ visualization/data/cima_correlations.json
├─ visualization/data/inflammation_disease.json
├─ visualization/data/validation_cima.json
└─ visualization/data/cross_atlas_*.json
```

### 3.2 From JSON to REST API Response

```
Client Request: GET /api/v1/cima/correlations?gene=IL17A&protein=IL17A

↓

Router (cima.py)
├─ Validate query parameters
└─ Call service method

↓

Service (cima_service.py)
├─ Check cache: correlation_data_cache
├─ If miss:
│  ├─ Load visualization/data/cima_correlations.json
│  ├─ Cache in-memory or Redis
│  └─ Return cached reference
├─ Filter by gene parameter
├─ Apply pagination (offset, limit)
└─ Return Pydantic response model

↓

Schema (cima.py)
├─ CorrelationResult (gene, protein, rho, p_value, count)
└─ Validated via Pydantic v2

↓

Response
├─ Content-Type: application/json
├─ Cache-Control headers (1 hour)
└─ Return to client
```

---

## 4. Technology Stack

### 4.1 Analysis Pipeline

| Layer | Technologies |
|-------|--------------|
| **Data Format** | H5AD (AnnData), CSV, Parquet (future) |
| **Data Processing** | Python 3.10+, NumPy, Pandas, SciPy |
| **GPU Acceleration** | CuPy (NVIDIA CUDA), with NumPy fallback |
| **Statistical Methods** | Spearman correlation, Wilcoxon rank-sum, ridge regression |
| **HPC** | SLURM (NIH Biowulf), bash scripting |
| **Signature Loading** | secactpy package (ridge regression implementation) |

### 4.2 REST API

| Layer | Technologies |
|-------|--------------|
| **Framework** | FastAPI (async/await) |
| **Validation** | Pydantic v2 |
| **HTTP Server** | Uvicorn (ASGI) |
| **Database** | PostgreSQL + SQLAlchemy (optional, not yet used) |
| **Cache** | Redis or in-memory (fallback) |
| **Authentication** | JWT (scaffolding), API keys |
| **Monitoring** | Logging, health check endpoints |

### 4.3 Web Frontend

| Layer | Technologies |
|-------|--------------|
| **Framework** | Vanilla JavaScript (no build step) |
| **Routing** | Client-side hash routing |
| **Visualization** | Plotly.js, D3.js v7 |
| **Data Format** | JSON |
| **Styling** | CSS3 (responsive, mobile-first) |
| **Browser Support** | Modern browsers (Chrome, Firefox, Safari) |

### 4.4 DevOps

| Component | Technology |
|-----------|-----------|
| **Containerization** | Docker, Singularity (HPC) |
| **Orchestration** | SLURM (HPC), docker-compose (local) |
| **Reverse Proxy** | Nginx |
| **Secrets** | Environment variables, .env files |

---

## 5. API Architecture

### 5.1 Router Structure (14 routers, 188+ endpoints)

```
/api/v1/
├── /health                    # 2 endpoints (health check, readiness)
├── /auth                      # 4 endpoints (login, register, verify, logout)
├── /cima/                     # ~32 endpoints
│  ├─ /summary                 # Atlas overview
│  ├─ /cell-types              # Available cell types
│  ├─ /correlations            # Age/BMI/biochemistry correlations
│  ├─ /differential            # Disease vs. healthy
│  └─ /eqtl                    # Genetic regulation (6 endpoints)
├── /inflammation/             # ~44 endpoints
│  ├─ /diseases                # List available diseases
│  ├─ /disease-activity        # Disease-specific activity patterns
│  ├─ /treatment-response      # Treatment response prediction
│  ├─ /cohort-validation       # Validation cohort consistency
│  └─ ... (10+ more endpoints)
├── /scatlas/                  # ~36 endpoints
│  ├─ /organs                  # Organ signatures
│  ├─ /cancer-types            # Cancer comparison
│  ├─ /immune-infiltration     # Immune cell distribution
│  ├─ /t-cell-exhaustion       # Exhaustion markers
│  └─ ... (8+ more endpoints)
├── /cross-atlas/              # ~28 endpoints
│  ├─ /summary                 # Atlas-level comparison
│  ├─ /conserved-signatures    # Signatures consistent across atlases
│  ├─ /cell-type-mapping       # Cell type harmonization
│  └─ ... (5+ more endpoints)
├── /validation/               # ~12 endpoints
│  ├─ /sample-level/{atlas}/{signature}       # Type 1
│  ├─ /celltype-level/{atlas}/{signature}     # Type 2
│  ├─ /pseudobulk-vs-singlecell/{atlas}/{sig} # Type 3
│  ├─ /singlecell-direct/{atlas}/{signature}  # Type 4
│  └─ /biological-associations/{atlas}        # Type 5
├── /search/                   # ~4 endpoints (global search)
├── /export/                   # ~6 endpoints (CSV, JSON export)
├── /chat/                     # ~4 endpoints (Claude AI conversation)
├── /submit/                   # ~4 endpoints (dataset submission)
├── /atlases/                  # ~6 endpoints (dynamic atlas registration)
└── /ws/                       # ~2 endpoints (WebSocket streaming)
```

### 5.2 Service Layer Pattern

```
Router (validates & routes)
  ↓
Service (business logic)
  ├─ Check cache
  ├─ Load data (JSON, Parquet)
  ├─ Filter/transform
  ├─ Compute statistics
  └─ Return Pydantic models
  ↓
Schema (Pydantic response model)
  ├─ Field validation
  ├─ Type conversion
  └─ JSON serialization
```

**Services**:
- `cima_service.py` - CIMA-specific logic
- `inflammation_service.py` - Inflammation atlas logic
- `scatlas_service.py` - scAtlas logic
- `cross_atlas_service.py` - Cross-atlas comparisons
- `validation_service.py` - 5-type validation framework
- `search_service.py` - Global search indexing
- `chat_service.py` - Claude API integration
- `base_service.py` - Shared utilities (JSON loading, caching)

### 5.3 Middleware Stack

1. **CORS**: Allow cross-origin requests (configurable)
2. **Rate Limiting**: Request throttling per IP/user
3. **Request Logging**: Log all requests/responses
4. **Error Handling**: Standardized error responses
5. **Authentication**: JWT token validation
6. **Compression**: GZIP response compression

---

## 6. Chat System Architecture (Round 3)

### 6.1 Modular Chat Package

Located in `cytoatlas-api/app/services/chat/`:

```
app/services/chat/
├── __init__.py              # Package exports
├── llm_client.py            # LLM backend abstraction (vLLM, Anthropic)
├── rag_service.py           # Retrieval-Augmented Generation
├── tool_executor.py         # Tool calling for API integration
└── conversation_service.py  # Conversation state management
```

### 6.2 Chat Flow

```
User Message
    ↓
Rate Limit Check
├─ Anonymous: 5 messages/day
└─ Authenticated: 1000 messages/day
    ↓
RAG Service (Retrieval)
├─ Embed query: all-MiniLM-L6-v2
├─ Search semantic DB: FAISS or SQLite
├─ Retrieve top-K documents (~175MB validation JSON)
└─ Return context chunks
    ↓
LLM Client (Primary: vLLM)
├─ Build prompt: system + context + query
├─ Call OpenAI-compatible endpoint (port 8001)
├─ Stream response
└─ Fallback to Anthropic if vLLM unavailable
    ↓
Tool Executor (Optional)
├─ Parse tool calls (function_calls)
├─ Execute API endpoints
├─ Return results to LLM
└─ Stream to user
    ↓
Conversation Service (Storage)
├─ Store in PostgreSQL (if available)
├─ Or in-memory with TTL cache
└─ Enable multi-turn conversations
    ↓
Response
├─ Stream SSE (for real-time)
└─ Or batch response
```

### 6.3 LLM Abstraction

```python
# app/services/chat/llm_client.py
class LLMClient(Protocol):
    async def chat(self, messages: List[Message]) -> str:
        """Call LLM with messages, return response."""

class VLLMClient(LLMClient):
    """OpenAI-compatible vLLM server (primary)"""
    def __init__(self, base_url: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    async def chat(self, messages):
        response = await self.client.chat.completions.create(...)
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    """Anthropic Claude (fallback)"""
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    async def chat(self, messages):
        response = await self.client.messages.create(...)
        return response.content[0].text
```

**Key design**: Protocol-based abstraction allows testing with mock client, fallback on vLLM failure.

### 6.4 RAG (Retrieval-Augmented Generation)

```python
# app/services/chat/rag_service.py
class RAGService:
    def __init__(self, db_path: str, model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model)
        self.vector_db = FAISS.load_local(db_path)  # Pre-built from validation JSON

    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Embed query, search, return top-K documents."""
        query_embedding = self.embedder.encode(query)
        scores, indices = self.vector_db.search(query_embedding, top_k)
        return [self.documents[i] for i in indices]

# Documents extracted from validation JSON files:
# - Disease activity patterns (from inflammation_disease_activity.json)
# - Organ signatures (from scatlas_organ_signatures.json)
# - Cross-atlas comparisons (from cross_atlas_conserved.json)
# - Validation metrics (from validation_*.json)
```

**Semantic database**: Pre-computed embeddings of all JSON files (~175-336MB per atlas).
**Update frequency**: Regenerated when JSON files change (via `06_preprocess_viz_data.py`).

### 6.5 Tool Calling

```python
# app/services/chat/tool_executor.py
TOOLS = [
    {
        "name": "search_genes",
        "description": "Search for genes across atlases",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"enum": ["gene", "cytokine", "protein", "all"]}
            }
        }
    },
    {
        "name": "get_disease_activity",
        "description": "Get cytokine activity for a disease",
        "parameters": {
            "disease": {"type": "string"},
            "signature_type": {"enum": ["CytoSig", "SecAct"]}
        }
    }
]

async def execute_tool(tool_name: str, **args) -> str:
    """Execute LLM-requested tool."""
    if tool_name == "search_genes":
        return await search_service.search(args["query"], args.get("type", "all"))
    elif tool_name == "get_disease_activity":
        return await inflammation_service.get_disease_activity(**args)
```

---

## 7. Frontend Architecture (Round 3)

### 7.1 Single-Page Application (SPA)

Located in `cytoatlas-api/static/`:

```
static/
├── index.html                # SPA shell
├── js/
│  ├── app.js                # Main orchestrator
│  ├── router.js             # Client-side routing
│  ├── api.js                # Fetch wrapper + caching
│  ├── pages/                # Page components (8 total)
│  │  ├── landing.js
│  │  ├── explore.js
│  │  ├── compare.js
│  │  ├── validate.js
│  │  ├── chat.js
│  │  ├── submit.js
│  │  ├── about.js
│  │  └── contact.js
│  └── components/           # Reusable UI components (20+)
│     ├── chart/             # Chart components
│     │  ├── LineChart.js
│     │  ├── ScatterChart.js
│     │  ├── HeatmapChart.js
│     │  ├── ViolinChart.js
│     │  └── BoxChart.js
│     ├── table/             # Table components
│     ├── modal/             # Dialog/modal components
│     └── form/              # Form components
└── css/
   ├── style.css             # Main styles
   ├── responsive.css        # Mobile/tablet
   └── dark-theme.css        # Dark mode (optional)
```

### 7.2 Chart Components

Each chart component wraps Plotly.js or D3.js with CytoAtlas-specific styling:

```javascript
// components/chart/LineChart.js
class LineChart {
  constructor(containerId, config) {
    this.containerId = containerId;
    this.config = config;
  }

  async render(data) {
    // data: {x: [], y: [], name: string, ...}
    const trace = {
      x: data.x,
      y: data.y,
      type: 'scatter',
      mode: 'lines+markers',
      name: data.name,
      marker: { size: 8, color: data.color || '#1f77b4' }
    };
    Plotly.newPlot(this.containerId, [trace], this.config.layout);
  }
}

// Usage in page
const chart = new LineChart('chart-container', {
  layout: { title: 'Age Correlation', xaxis: { title: 'Age (years)' } }
});
await chart.render(await api.get('/api/v1/cima/correlations/age'));
```

### 7.3 State Management

No external state library (keep dependencies minimal). Use module-level state:

```javascript
// app/state.js
const AppState = {
  currentAtlas: "CIMA",
  currentView: "explore",
  selectedCellType: null,
  selectedSignature: "CytoSig",
  userData: null,

  set: (key, value) => {
    AppState[key] = value;
    // Publish state change
    document.dispatchEvent(new CustomEvent('statechange', { detail: { key, value } }));
  },

  get: (key) => AppState[key]
};

// Listen for state changes
document.addEventListener('statechange', (e) => {
  const { key, value } = e.detail;
  console.log(`State changed: ${key} = ${value}`);
  // Re-render if needed
});
```

### 7.4 Data Loader with Caching

```javascript
// api.js
class APIClient {
  constructor() {
    this.cache = new Map();
    this.cacheTTL = 3600000; // 1 hour
  }

  async get(url, options = {}) {
    const cacheKey = url;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.data;
    }

    const response = await fetch(url, options);
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    this.cache.set(cacheKey, { data, timestamp: Date.now() });
    return data;
  }

  invalidateCache(pattern) {
    for (const [key] of this.cache) {
      if (key.includes(pattern)) this.cache.delete(key);
    }
  }
}

// Usage
const api = new APIClient();
const cimaActivity = await api.get('/api/v1/cima/activity?signature_type=CytoSig');
```

---

## 8. Pipeline Management (Round 3)

### 8.1 Pipeline Architecture

Located in `cytoatlas-api/app/routers/pipeline.py`:

```
Pipeline Management System
├─ Pipeline Registry (available pipelines)
├─ Dependency Graph (task ordering)
├─ Execution Engine (run tasks)
├─ Output Validator (verify results)
└─ Status Tracker (monitor progress)
```

### 8.2 Dependency Graph

Analysis pipelines follow strict order:

```
00_pilot_analysis.py          (Validate 100K subset)
    ↓
01_cima_activity.py           (CIMA: 6.5M cells)
02_inflam_activity.py         (Inflammation: 6.3M cells)
03_scatlas_analysis.py        (scAtlas: 6.4M cells)
    ↓
04_integrated.py              (Cross-atlas comparison)
    ↓
05_figures.py                 (Publication plots)
06_preprocess_viz_data.py     (Generate JSON for web)
    ↓
API Ready (all endpoints functional)
```

### 8.3 Pipeline Status Endpoint

```bash
GET /api/v1/pipeline/status

Response:
{
  "status": "running",              # idle, running, completed, failed
  "current_step": "02_inflam_activity.py",
  "progress": 45,                   # 0-100%
  "start_time": "2026-02-09T08:00:00Z",
  "estimated_completion": "2026-02-09T14:30:00Z",
  "tasks": [
    {
      "name": "00_pilot_analysis.py",
      "status": "completed",
      "duration_minutes": 120,
      "output_files": ["results/pilot/..."]
    },
    {
      "name": "01_cima_activity.py",
      "status": "running",
      "progress": 65,
      "eta_minutes": 240
    }
  ]
}
```

### 8.4 Result Validation

After each pipeline step:

1. **File Existence Check**: All expected outputs generated?
2. **Data Integrity**: Row counts match expectations? No NaN explosion?
3. **Lineage Verification**: Output files traceable to input files?
4. **Statistical Sanity**: Activity scores in [-3, +3] range? Correlations in [-1, +1]?
5. **Cross-Check**: Results consistent with previous step?

---

## 9. Data Layer Architecture

### 6.1 Data Storage Layers

```
Tier 1: Hot Data (In-Memory Cache)
├─ Frequently accessed results (cima_correlations, disease_activity)
├─ TTL: 1 hour
├─ Medium: Redis (or in-memory dict fallback)
├─ Size: ~100MB typical
└─ Hit rate target: >80%

Tier 2: Warm Data (JSON Files)
├─ Pre-computed results (visualization/data/*.json)
├─ Medium: Local filesystem
├─ Size: ~500MB total
└─ Access pattern: On-demand, cached in Tier 1

Tier 3: Cold Data (Results Directory)
├─ Raw CSV analysis outputs (results/*/atlas*.csv)
├─ Medium: Network storage (/vf/users/...)
├─ Size: ~50GB
└─ Access pattern: Rare (used for data validation, re-generation)
```

### 9.2 Repository Pattern (Implemented Framework)

Future data abstraction layer for testability and backend swappability:

```python
# Protocol-based abstraction
class DataRepository(Protocol):
    async def get_correlations(self, gene: str) -> CorrelationData: ...
    async def get_disease_activity(self, disease: str) -> ActivityData: ...

# Implementations
class JSONRepository(DataRepository):
    # Load from visualization/data/*.json

class ParquetRepository(DataRepository):
    # Load from results/*.parquet (future)

class PostgreSQLRepository(DataRepository):
    # Query from database tables
```

---

## 10. Security Architecture (Round 2-3)

### 10.1 RBAC Model (Implemented)

Five-role model planned for Round 2:

| Role | Permissions |
|------|-------------|
| **anonymous** | Read public data, search, basic API endpoints |
| **viewer** | Read all public datasets, access dashboard |
| **researcher** | Download data, access advanced analytics |
| **data_curator** | Submit custom datasets, manage metadata |
| **admin** | System administration, user management, audit logs |

### 10.2 Authentication Flow

```
1. User Login
   POST /api/v1/auth/login { username, password }
   ↓
2. JWT Token Issued
   Response: { access_token, refresh_token, expires_in }
   ↓
3. Requests Authenticated
   Header: Authorization: Bearer {access_token}
   ↓
4. Token Validation
   Middleware: Verify JWT signature, check expiration
   ↓
5. Role-Based Access
   Route: Check user.role against required permissions
```

### 10.3 Audit Logging (Implemented Framework)

All data access logged to JSONL file with context:
- Timestamp, user ID, email, IP address
- HTTP method, endpoint, status code
- Dataset accessed, action (read/write/export)
- Retention: 90 days in DB, 30 days in logs

Configuration in `app/config.py`:
- `audit_enabled`: Enable/disable audit logging
- `audit_log_path`: Path to audit log file (default: `logs/audit.jsonl`)

### 10.4 Prompt Injection Defense

Chat system validates all LLM inputs:

1. **RAG Grounding**: Responses must cite retrieved documents
2. **System Prompt Enforcement**: LLM constrained to CytoAtlas domain
3. **Tool Validation**: Tool parameters validated before execution
4. **Rate Limiting**: Per-user/session message throttling
5. **Output Sanitization**: HTML/script tags removed from responses

---

## 11. Deployment Architecture

### 11.1 Development Mode (Local)

```bash
# Terminal 1: Start API
cd cytoatlas-api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: View dashboard
# Open http://localhost:8000/
```

### 11.2 HPC (SLURM) Mode

```bash
# Submit unified job (vLLM + API)
sbatch scripts/slurm/run_vllm.sh

# Job components:
# - vLLM inference server (GPU, for chat)
# - Uvicorn API server (CPU)
# - Health check orchestration

# Result:
# - API running on compute node
# - Accessible via SSH tunnel or proxy
```

### 11.3 Production Mode (Docker)

```bash
# Build image
docker build -t cytoatlas-api:latest .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  cytoatlas-api:latest

# Behind Nginx reverse proxy
# - SSL/TLS termination
# - Load balancing
# - Static file serving
```

---

## 12. Domain-Driven Design (DDD) Roadmap

### 12.1 Current Bounded Contexts

```
┌─────────────────────────────────────────────────────────┐
│  Activity Analysis Bounded Context                       │
│  ├─ Entity: Signature (CytoSig, SecAct, LinCytoSig)     │
│  ├─ Entity: Activity (per sample, per cell type)        │
│  ├─ Value Object: ActivityScore (z-score, CI)          │
│  └─ Aggregate: AtlasAnalysis (all signatures for atlas) │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Validation Bounded Context                              │
│  ├─ Entity: ValidationMetric                             │
│  ├─ Value Object: Credibility (5-type assessment)       │
│  └─ Aggregate: AtlasValidation                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Data Access Bounded Context                             │
│  ├─ Interface: DataRepository                            │
│  ├─ Service: CacheManager                               │
│  └─ Factory: RepositoryFactory                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  User Submission Bounded Context                         │
│  ├─ Entity: UserAtlas                                   │
│  ├─ Service: SubmissionService                          │
│  └─ Factory: AtlasProcessingFactory                     │
└─────────────────────────────────────────────────────────┘
```

### 12.2 Anti-Corruption Layers (Planned for Round 4)

```
External Data Source (e.g., cellxgene)
       ↓
Anti-Corruption Layer (Adapter)
├─ Map external format → CytoAtlas schema
├─ Validate data quality
└─ Store in standardized format
       ↓
User Submission Bounded Context
├─ Process standardized data
├─ Run activity inference
└─ Generate validation metrics
```

### 12.3 Evolution Path (Rounds 1-4)

| Round | Milestone | Status | Key Achievements |
|-------|-----------|--------|------------------|
| Round 1 | Documentation Cleanup | ✅ Complete | ARCHITECTURE.md, ADRs, archive system |
| Round 2 | Security Hardening | ✅ Complete | JWT, RBAC, audit logging, rate limiting |
| Round 3 | Data Layer & Documentation | ✅ Complete | Chat system, frontend components, pipeline management, user docs |
| Round 4 | Extensibility (Planned) | Pending | User submissions, API v2, GraphQL |

---

## 13. Deployment Checklist

### 13.1 Development Deployment

- [x] FastAPI application created
- [x] All 14 routers implemented
- [x] All 12 services implemented
- [x] Frontend SPA completed (8 pages)
- [x] 30+ JSON data files generated
- [x] Local testing with Uvicorn

### 13.2 HPC Deployment

- [x] SLURM job script (`run_vllm.sh`)
- [x] vLLM + API unified orchestration
- [x] Health check monitoring
- [x] Network proxy configuration
- [ ] Performance benchmarking (10K+ req/s)

### 13.3 Production Hardening (Round 2-3)

- [ ] JWT authentication
- [ ] RBAC role enforcement
- [ ] Rate limiting per user/IP
- [ ] Audit logging
- [ ] Prometheus metrics
- [ ] Load testing (k6, locust)
- [ ] SSL/TLS certificates
- [ ] WAF (Web Application Firewall) rules

---

## 14. Monitoring & Operations

### 14.1 Health Check Endpoints

```
GET /api/v1/health              # Basic liveness
GET /api/v1/health/ready        # Readiness (all services available)
```

### 14.2 Key Metrics to Monitor

| Metric | Target | Tool |
|--------|--------|------|
| API response time | <200ms p95 | Prometheus |
| Request throughput | >1000 req/s | Prometheus |
| Cache hit rate | >80% | Custom counters |
| Error rate | <0.1% | Application logs |
| Database query time | <100ms | SQLAlchemy logging |

### 14.3 Alerting (Future)

```
Alert: API response time > 500ms
Alert: Cache miss rate > 20%
Alert: Error rate > 1%
Alert: Database connection pool exhausted
```

---

## 15. Documentation References

### 15.1 Key Documents

| Document | Location | Purpose |
|----------|----------|---------|
| **Project Overview** | [CLAUDE.md](CLAUDE.md) | Quick reference, TODOs, lessons learned |
| **Dataset Details** | [datasets/README.md](datasets/README.md) | Data source specifications |
| **Pipeline Documentation** | [pipelines/README.md](pipelines/README.md) | Analysis pipeline details |
| **Output Catalog** | [outputs/README.md](outputs/README.md) | File structure and lineage |
| **Deployment Guide** | [docs/DEPLOYMENT.md](DEPLOYMENT.md) | HPC/SLURM setup, environment variables |
| **API Reference** | [docs/API_REFERENCE.md](API_REFERENCE.md) | All 188+ endpoints with examples |
| **User Guide** | [docs/USER_GUIDE.md](USER_GUIDE.md) | Using CytoAtlas (atlases, chat, exports) |
| **Decisions Log** | [decisions/README.md](decisions/README.md) | Architecture Decision Records (ADRs) |
| **Archived Plans** | [archive/README.md](archive/README.md) | Historical planning documents |

### 15.2 Machine-Readable Metadata

- `docs/registry.json` - File catalog with lineage, types, APIs
- `docs/EMBEDDED_DATA_CHECKLIST.md` - JSON files required in frontend

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Activity** | Ridge regression-based inference of signature expression (z-score) |
| **Activity Difference** | Simple difference of activities between groups (not log2 fold-change) |
| **Pseudo-bulk** | Aggregated expression by sample × cell type (primary analysis level) |
| **Signature** | Gene set with weights (CytoSig 44 cytokines, SecAct 1,249 proteins) |
| **Bounded Context** | Explicit boundary defining a domain model (DDD) |
| **Repository Pattern** | Abstraction layer for data access (planned) |
| **RBAC** | Role-Based Access Control (5 roles planned for Round 2) |
| **ADR** | Architecture Decision Record (document rationale for design choices) |

---

## 17. Contact & Support

For questions or issues:
1. Check [CLAUDE.md](CLAUDE.md) for common solutions
2. Review [Lessons Learned](CLAUDE.md#lessons-learned) section
3. Consult relevant ADR in [decisions/](decisions/) directory
4. Examine archived plans in [archive/plans/](archive/plans/)
