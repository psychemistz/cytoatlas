# Chat Service Architecture

Refactored chat system for CytoAtlas with modular, composable components.

## Components

### 1. LLM Client Layer (`llm_client.py`)

Unified interface for multiple LLM backends:

- **`VLLMClient`**: OpenAI-compatible client for vLLM
- **`AnthropicClient`**: Anthropic Claude API client
- **`DualLLMClient`**: Tries vLLM first, falls back to Anthropic on connection error

**Features:**
- Automatic JSON repair for malformed tool arguments (Mistral quirk)
- Streaming and non-streaming support
- Consistent response format across backends

### 2. Tool System (`tool_definitions.py`, `tool_executor.py`)

Tool execution layer with enhanced capabilities:

- **`tool_definitions.py`**: Centralized tool schemas (imports from `mcp_tools.py`)
- **`ToolExecutor`**: Executes tools with result chunking and format conversion
- **`ToolResultChunker`**: Truncates large results (>4000 chars) to prevent context overflow
- **`ToolCallSerializer`**: Converts between Anthropic and OpenAI tool formats

### 3. RAG Service (`rag_service.py`, `embeddings.py`)

Semantic search over documentation and context:

- **`EmbeddingService`**: Sentence-transformers embeddings (384-dim, CPU-only)
- **`RAGService`**: LanceDB vector search
- **`RAGResult`**: Structured search results with source tracking

**Indexed Content:**
1. Documentation chunks (`docs/*.md`)
2. Column definitions (from `registry.json`)
3. Atlas summaries (CIMA, Inflammation, scAtlas)
4. Biological context (cytokines, cell types, known biology)
5. Data summaries (`*_summary.json` files)

### 4. Conversation Persistence (`conversation_service.py`)

Conversation and message storage:

- **`ConversationService`**: Manages conversations and messages
- SQLAlchemy async support (with in-memory fallback)
- Data caching for downloads
- Session-based and user-based conversation tracking

### 5. Chat Orchestrator (`chat_service.py`)

Main service that composes all components:

- RAG-enhanced prompts
- Tool execution loop
- Conversation management
- Citation tracking

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     ChatService                              │
│  (Orchestrates all components)                               │
└───┬──────────────┬──────────────┬──────────────┬───────────┘
    │              │              │              │
    ▼              ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐
│ LLM     │  │ RAG      │  │ Tool     │  │ Conversation     │
│ Client  │  │ Service  │  │ Executor │  │ Service          │
└─────────┘  └──────────┘  └──────────┘  └──────────────────┘
    │              │              │              │
    ▼              ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐
│ vLLM /  │  │ LanceDB  │  │ MCP      │  │ SQLAlchemy /     │
│ Claude  │  │ + s-t    │  │ Tools    │  │ In-Memory        │
└─────────┘  └──────────┘  └──────────┘  └──────────────────┘
```

## Usage

### Basic Chat

```python
from app.services.chat import get_chat_service

chat_service = get_chat_service()

response = await chat_service.chat(
    content="What is the activity of IFNG in CD8 T cells?",
    session_id="user-session-123",
)

print(response.content)
print(response.citations)  # RAG sources
```

### Streaming Chat

```python
async for chunk in chat_service.chat_stream(
    content="Compare TNF activity across atlases",
    session_id="user-session-123",
):
    if chunk.type == "text":
        print(chunk.content, end="", flush=True)
    elif chunk.type == "tool_call":
        print(f"\nCalling tool: {chunk.tool_call.name}")
    elif chunk.type == "visualization":
        render_viz(chunk.visualization)
```

## Configuration

Add to `.env`:

```bash
# LLM Configuration
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=not-needed
CHAT_MODEL=mistralai/Mistral-Small-3.1-24B-Instruct-2503
ANTHROPIC_API_KEY=sk-ant-...  # Fallback

# RAG Configuration
RAG_ENABLED=true
RAG_DB_PATH=rag_db
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_TOP_K=5
```

## Building the RAG Index

```bash
# From project root
python scripts/build_rag_index.py --db-path cytoatlas-api/rag_db

# This will:
# 1. Load sentence-transformers model
# 2. Extract ~30-50 docs chunks from docs/
# 3. Extract ~15 column definitions
# 4. Extract 3 atlas summaries
# 5. Extract ~15 biological context entries
# 6. Embed all documents
# 7. Create LanceDB index
```

## Migration from Legacy

The refactored `ChatService` maintains the same interface as the legacy version:

```python
# Old code (chat_service_legacy.py)
from app.services.chat_service import get_chat_service

# New code (chat/chat_service.py)
from app.services.chat import get_chat_service

# Same interface!
service = get_chat_service()
response = await service.chat(content="...", session_id="...")
```

**Zero router changes needed** - the chat router continues to work unchanged.

## Dependencies

Added to `pyproject.toml`:

```toml
"lancedb>=0.4.0",
"sentence-transformers>=2.2.0",
```

Install:

```bash
pip install -e ".[dev]"
```

## Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Composability**: Components can be swapped or mocked for testing
3. **Backward Compatibility**: Existing API contracts preserved
4. **Lazy Loading**: Heavy dependencies (models, DB) loaded on first use
5. **Graceful Degradation**: RAG can be disabled, falls back to tool-only mode

## Testing

```python
# Unit test with mocks
chat_service = ChatService(
    llm_client=MockLLMClient(),
    rag_service=MockRAGService(),
    conversation_service=MockConversationService(),
    tool_executor=MockToolExecutor(),
)

# Integration test with real components
chat_service = get_chat_service(rag_enabled=False)
```

## Performance

- **Embeddings**: CPU-only, ~100 docs/sec on HPC node
- **RAG Search**: <50ms for top-5 results
- **Tool Execution**: Variable (depends on data size)
- **LLM Latency**:
  - vLLM: ~500ms first token, ~50 tokens/sec
  - Claude: ~1s first token, ~40 tokens/sec

## Future Enhancements

- [ ] Database persistence for conversations (currently in-memory)
- [ ] Advanced RAG: reranking, hybrid search
- [ ] Tool result caching
- [ ] Multi-turn tool execution planning
- [ ] User feedback on citations
