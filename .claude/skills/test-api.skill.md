# test-api

Test CytoAtlas API endpoints

## Instructions

When the user invokes /test-api, perform the following:

1. **Start the API server** (if not running):
   ```bash
   cd /data/parks34/projects/2cytoatlas/cytoatlas-api
   source ~/bin/myconda && conda activate secactpy
   uvicorn app.main:app --host 0.0.0.0 --port 8000 &
   sleep 3
   ```

2. **Test endpoints based on arguments**:
   - `/test-api health` - Test /health and /api/health endpoints
   - `/test-api search <gene>` - Test search endpoints for a gene
   - `/test-api cima` - Test CIMA router endpoints
   - `/test-api inflam` - Test Inflammation router endpoints
   - `/test-api scatlas` - Test scAtlas router endpoints
   - `/test-api perturbation` - Test Perturbation router endpoints (parse10m + tahoe)
   - `/test-api spatial` - Test Spatial router endpoints
   - `/test-api all` - Run all endpoint tests

3. **Report results** with:
   - Response status codes
   - Response times
   - Sample data shape (for JSON responses)
   - Any errors encountered

4. **Clean up**: Stop the server if it was started by this skill

## Example Usage

```
/test-api search IFNG
/test-api cima
/test-api all
```
