#!/usr/bin/env python3
"""
Documentation Generator for CytoAtlas Project
==============================================
Semi-automated doc generator that:
- Extracts function names and docstrings from preprocessing scripts
- Inspects H5AD files for column metadata
- Analyzes JSON files for schema inference
- Generates markdown scaffolds for manual curation
- Inventories API endpoints by router
- Catalogs Pydantic schemas
"""

import os
import sys
import ast
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re

# Paths
PROJECT_ROOT = Path("/data/parks34/projects/2cytoatlas")
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VIZ_DATA_DIR = PROJECT_ROOT / "visualization" / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_functions_from_script(script_path: Path) -> List[Dict[str, Any]]:
    """Extract function definitions and docstrings from a Python script."""
    functions = []

    try:
        with open(script_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'lineno': node.lineno,
                    'docstring': ast.get_docstring(node) or '',
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d)
                                   for d in node.decorator_list]
                }
                functions.append(func_info)
    except Exception as e:
        print(f"  Error parsing {script_path}: {e}")

    return functions


def extract_script_docstring(script_path: Path) -> str:
    """Extract module-level docstring from a Python script."""
    try:
        with open(script_path, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        return ast.get_docstring(tree) or ''
    except Exception:
        return ''


def inspect_h5ad_metadata(h5ad_path: Path) -> Dict[str, Any]:
    """Inspect an H5AD file and extract column metadata."""
    metadata = {
        'path': str(h5ad_path),
        'exists': h5ad_path.exists(),
        'obs_columns': [],
        'var_columns': [],
        'n_obs': 0,
        'n_vars': 0,
        'layers': [],
        'size_bytes': 0
    }

    if not h5ad_path.exists():
        return metadata

    try:
        import anndata as ad

        # Get file size
        metadata['size_bytes'] = h5ad_path.stat().st_size

        # Load in backed mode for efficiency
        adata = ad.read_h5ad(h5ad_path, backed='r')

        metadata['n_obs'] = adata.n_obs
        metadata['n_vars'] = adata.n_vars

        # Extract obs columns with sample values
        for col in adata.obs.columns:
            col_info = {
                'name': col,
                'dtype': str(adata.obs[col].dtype),
                'n_unique': adata.obs[col].nunique() if hasattr(adata.obs[col], 'nunique') else 'N/A',
                'sample_values': []
            }

            # Get sample values
            try:
                unique_vals = adata.obs[col].dropna().unique()[:5]
                col_info['sample_values'] = [str(v) for v in unique_vals]
            except Exception:
                pass

            metadata['obs_columns'].append(col_info)

        # Extract var columns
        for col in adata.var.columns:
            col_info = {
                'name': col,
                'dtype': str(adata.var[col].dtype)
            }
            metadata['var_columns'].append(col_info)

        # Check for layers
        if hasattr(adata, 'layers') and adata.layers:
            metadata['layers'] = list(adata.layers.keys())

        adata.file.close()

    except Exception as e:
        metadata['error'] = str(e)

    return metadata


def inspect_csv_metadata(csv_path: Path, max_rows: int = 5) -> Dict[str, Any]:
    """Inspect a CSV file and extract column metadata."""
    import pandas as pd

    metadata = {
        'path': str(csv_path),
        'exists': csv_path.exists(),
        'columns': [],
        'n_rows': 0,
        'size_bytes': 0
    }

    if not csv_path.exists():
        return metadata

    try:
        metadata['size_bytes'] = csv_path.stat().st_size

        df = pd.read_csv(csv_path, nrows=1000)  # Sample for inspection
        metadata['n_rows'] = len(df)

        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'n_unique': df[col].nunique(),
                'sample_values': [str(v) for v in df[col].dropna().unique()[:5]]
            }
            metadata['columns'].append(col_info)

    except Exception as e:
        metadata['error'] = str(e)

    return metadata


def inspect_json_schema(json_path: Path) -> Dict[str, Any]:
    """Infer schema from a JSON file."""
    metadata = {
        'path': str(json_path),
        'exists': json_path.exists(),
        'type': None,
        'fields': [],
        'size_bytes': 0,
        'n_records': 0
    }

    if not json_path.exists():
        return metadata

    try:
        metadata['size_bytes'] = json_path.stat().st_size

        with open(json_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            metadata['type'] = 'array'
            metadata['n_records'] = len(data)

            if len(data) > 0 and isinstance(data[0], dict):
                # Infer fields from first record
                sample = data[0]
                for key, value in sample.items():
                    field_info = {
                        'name': key,
                        'type': type(value).__name__,
                        'sample': str(value)[:50] if value is not None else None
                    }
                    metadata['fields'].append(field_info)

        elif isinstance(data, dict):
            metadata['type'] = 'object'

            for key, value in data.items():
                field_info = {
                    'name': key,
                    'type': type(value).__name__
                }

                if isinstance(value, list):
                    field_info['n_items'] = len(value)
                    if len(value) > 0 and isinstance(value[0], dict):
                        field_info['item_fields'] = list(value[0].keys())

                metadata['fields'].append(field_info)

    except Exception as e:
        metadata['error'] = str(e)

    return metadata


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def extract_api_endpoints(router_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """Extract API endpoints from FastAPI router files."""
    endpoints_by_router = defaultdict(list)

    router_files = list(router_dir.glob("*.py"))

    for router_file in router_files:
        if router_file.name == "__init__.py":
            continue

        try:
            with open(router_file, 'r') as f:
                content = f.read()

            # Extract route decorators: @router.get/post/put/delete/patch(...)
            pattern = r'@router\.(get|post|put|delete|patch|options)\((["\']([^"\']+)["\'])'
            matches = re.findall(pattern, content)

            for method, full_path, path in matches:
                endpoints_by_router[router_file.stem].append({
                    'method': method.upper(),
                    'path': path
                })
        except Exception as e:
            print(f"  Warning: Failed to parse {router_file.name}: {e}")

    return dict(endpoints_by_router)


def extract_pydantic_schemas(schemas_dir: Path) -> Dict[str, List[str]]:
    """Extract Pydantic model definitions from schema files."""
    schemas_by_file = defaultdict(list)

    schema_files = list(schemas_dir.glob("*.py"))

    for schema_file in schema_files:
        if schema_file.name == "__init__.py":
            continue

        try:
            with open(schema_file, 'r') as f:
                source = f.read()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a Pydantic model (BaseModel subclass)
                    for base in node.bases:
                        if (isinstance(base, ast.Name) and 'Model' in base.id) or \
                           (isinstance(base, ast.Attribute) and 'BaseModel' in ast.unparse(base)):
                            schemas_by_file[schema_file.stem].append(node.name)
                            break
        except Exception as e:
            print(f"  Warning: Failed to parse {schema_file.name}: {e}")

    return dict(schemas_by_file)


def inventory_data_files(viz_data_dir: Path) -> Dict[str, Any]:
    """Catalog all visualization data files with metadata."""
    inventory = {
        'total_size': 0,
        'total_files': 0,
        'by_atlas': defaultdict(lambda: {'files': [], 'size': 0}),
        'largest_files': []
    }

    file_info = []

    for json_file in sorted(viz_data_dir.glob("**/*.json")):
        if json_file.is_file():
            size = json_file.stat().st_size

            # Determine atlas
            name = json_file.name
            if 'cima' in name.lower():
                atlas = 'cima'
            elif 'inflam' in name.lower():
                atlas = 'inflammation'
            elif 'scatlas' in name.lower() or 'cancer' in name.lower() or 'exhaustion' in name.lower():
                atlas = 'scatlas'
            elif 'cross_atlas' in name or 'validation' in name:
                atlas = 'cross_atlas'
            else:
                atlas = 'other'

            file_info.append({
                'name': name,
                'path': str(json_file.relative_to(viz_data_dir)),
                'size': size,
                'size_human': format_bytes(size),
                'atlas': atlas
            })

            inventory['total_size'] += size
            inventory['total_files'] += 1
            inventory['by_atlas'][atlas]['files'].append(name)
            inventory['by_atlas'][atlas]['size'] += size

    # Sort by size, keep top 10
    file_info.sort(key=lambda x: x['size'], reverse=True)
    inventory['largest_files'] = file_info[:10]

    return inventory


def generate_endpoint_inventory(endpoints: Dict[str, List[Dict[str, str]]]) -> str:
    """Generate markdown documentation for API endpoint inventory."""
    doc = "# CytoAtlas API Endpoint Inventory\n\n"
    doc += f"**Generated**: {__import__('datetime').datetime.now().isoformat()}\n\n"

    total_endpoints = sum(len(eps) for eps in endpoints.values())
    doc += f"**Total Endpoints**: {total_endpoints}\n"
    doc += f"**Total Routers**: {len(endpoints)}\n\n"

    # Summary table
    doc += "## Summary by Router\n\n"
    doc += "| Router | GET | POST | PUT | DELETE | PATCH | Total |\n"
    doc += "|--------|-----|------|-----|--------|-------|-------|\n"

    for router_name in sorted(endpoints.keys()):
        endpoints_list = endpoints[router_name]
        methods = defaultdict(int)
        for ep in endpoints_list:
            methods[ep['method']] += 1

        row = f"| {router_name} "
        for method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
            row += f"| {methods.get(method, 0)} "
        row += f"| {len(endpoints_list)} |\n"
        doc += row

    # Detail by router
    doc += "\n## Endpoints by Router\n\n"

    for router_name in sorted(endpoints.keys()):
        doc += f"### {router_name}\n\n"
        doc += "| Method | Path |\n"
        doc += "|--------|------|\n"

        for ep in endpoints[router_name]:
            doc += f"| {ep['method']} | `{ep['path']}` |\n"

        doc += "\n"

    return doc


def generate_schema_inventory(schemas: Dict[str, List[str]]) -> str:
    """Generate markdown documentation for Pydantic schema inventory."""
    doc = "# CytoAtlas Pydantic Schema Inventory\n\n"
    doc += f"**Generated**: {__import__('datetime').datetime.now().isoformat()}\n\n"

    total_schemas = sum(len(models) for models in schemas.values())
    doc += f"**Total Schemas**: {total_schemas}\n"
    doc += f"**Total Files**: {len(schemas)}\n\n"

    doc += "## Schemas by File\n\n"

    for file_name in sorted(schemas.keys()):
        models = schemas[file_name]
        doc += f"### {file_name}\n\n"

        for model_name in sorted(models):
            doc += f"- `{model_name}`\n"

        doc += f"\n({len(models)} schemas)\n\n"

    return doc


def generate_data_file_inventory(inventory: Dict[str, Any]) -> str:
    """Generate markdown documentation for data file inventory."""
    doc = "# CytoAtlas Data File Inventory\n\n"
    doc += f"**Generated**: {__import__('datetime').datetime.now().isoformat()}\n\n"

    doc += "## Summary\n\n"
    doc += f"| Metric | Value |\n"
    doc += f"|--------|-------|\n"
    doc += f"| Total Files | {inventory['total_files']} |\n"
    doc += f"| Total Size | {format_bytes(inventory['total_size'])} |\n\n"

    doc += "## Largest Files\n\n"
    doc += "| Rank | File | Size | Atlas |\n"
    doc += "|------|------|------|-------|\n"

    for i, f in enumerate(inventory['largest_files'], 1):
        doc += f"| {i} | `{f['name']}` | {f['size_human']} | {f['atlas']} |\n"

    doc += "\n## Files by Atlas\n\n"

    for atlas in sorted(inventory['by_atlas'].keys()):
        atlas_data = inventory['by_atlas'][atlas]
        doc += f"### {atlas}\n\n"
        doc += f"**Files**: {len(atlas_data['files'])}\n"
        doc += f"**Total Size**: {format_bytes(atlas_data['size'])}\n\n"

        for fname in sorted(atlas_data['files']):
            doc += f"- `{fname}`\n"

        doc += "\n"

    return doc


def generate_dataset_doc(atlas: str, h5ad_meta: Dict, csv_metas: List[Dict]) -> str:
    """Generate dataset documentation markdown."""

    doc = f"""# {atlas.upper()} Dataset

## Overview

| Property | Value |
|----------|-------|
| **Cells** | {h5ad_meta.get('n_obs', 'N/A'):,} |
| **Genes** | {h5ad_meta.get('n_vars', 'N/A'):,} |
| **File Size** | {format_bytes(h5ad_meta.get('size_bytes', 0))} |

## File Path

```python
H5AD_PATH = '{h5ad_meta.get('path', 'N/A')}'
```

## Cell Observations (`.obs`)

| Column | Type | Unique Values | Examples |
|--------|------|---------------|----------|
"""

    for col in h5ad_meta.get('obs_columns', []):
        examples = ', '.join(col.get('sample_values', [])[:3])
        doc += f"| `{col['name']}` | {col['dtype']} | {col.get('n_unique', 'N/A')} | {examples} |\n"

    if h5ad_meta.get('var_columns'):
        doc += """
## Gene Variables (`.var`)

| Column | Type | Description |
|--------|------|-------------|
"""
        for col in h5ad_meta.get('var_columns', []):
            doc += f"| `{col['name']}` | {col['dtype']} | |\n"

    if h5ad_meta.get('layers'):
        doc += f"""
## Layers

Available layers: {', '.join(h5ad_meta['layers'])}
"""

    # Add metadata files
    for csv_meta in csv_metas:
        if csv_meta.get('columns'):
            doc += f"""
## Metadata: {Path(csv_meta['path']).name}

| Column | Type | Unique Values | Examples |
|--------|------|---------------|----------|
"""
            for col in csv_meta['columns']:
                examples = ', '.join(col.get('sample_values', [])[:3])
                doc += f"| `{col['name']}` | {col['dtype']} | {col.get('n_unique', 'N/A')} | {examples} |\n"

    doc += """
## Usage Examples

```python
import anndata as ad

# Backed mode (memory efficient)
adata = ad.read_h5ad(H5AD_PATH, backed='r')

# Access cell type distribution
adata.obs['cell_type'].value_counts()
```
"""

    return doc


def generate_json_catalog(json_files: List[Dict]) -> str:
    """Generate JSON file catalog markdown."""

    doc = """# Visualization JSON Catalog

Complete catalog of JSON files for the web dashboard.

## Files

| File | Size | Type | Records | Description |
|------|------|------|---------|-------------|
"""

    for jf in sorted(json_files, key=lambda x: x.get('path', '')):
        name = Path(jf['path']).name
        size = format_bytes(jf.get('size_bytes', 0))
        jtype = jf.get('type', 'unknown')
        records = jf.get('n_records', '-')

        doc += f"| `{name}` | {size} | {jtype} | {records} | |\n"

    doc += """
## Schema Details

"""

    for jf in sorted(json_files, key=lambda x: x.get('path', '')):
        name = Path(jf['path']).name

        doc += f"""### {name}

| Field | Type | Sample |
|-------|------|--------|
"""
        for field in jf.get('fields', []):
            sample = field.get('sample', '')[:30] if field.get('sample') else ''
            doc += f"| `{field['name']}` | {field['type']} | {sample} |\n"

        doc += "\n"

    return doc


def generate_pipeline_doc(script_path: Path, functions: List[Dict], docstring: str) -> str:
    """Generate pipeline documentation markdown."""

    script_name = script_path.name

    doc = f"""# {script_name} Pipeline

## Overview

{docstring}

## Script Location

```
{script_path}
```

## Functions

| Function | Line | Description |
|----------|------|-------------|
"""

    for func in functions:
        # Get first line of docstring as description
        desc = func['docstring'].split('\n')[0] if func['docstring'] else ''
        doc += f"| `{func['name']}()` | {func['lineno']} | {desc} |\n"

    doc += """
## Function Details

"""

    for func in functions:
        if func['name'].startswith('_'):
            continue  # Skip private functions

        doc += f"""### {func['name']}

```python
def {func['name']}({', '.join(func['args'])})
```

{func['docstring']}

---

"""

    return doc


def main():
    parser = argparse.ArgumentParser(description='Generate CytoAtlas documentation')
    parser.add_argument('--datasets', action='store_true', help='Generate dataset docs')
    parser.add_argument('--pipelines', action='store_true', help='Generate pipeline docs')
    parser.add_argument('--json', action='store_true', help='Generate JSON catalog')
    parser.add_argument('--endpoints', action='store_true', help='Generate API endpoint inventory')
    parser.add_argument('--schemas', action='store_true', help='Generate Pydantic schema inventory')
    parser.add_argument('--data-files', action='store_true', help='Generate data file inventory')
    parser.add_argument('--all', action='store_true', help='Generate all docs')
    parser.add_argument('--inspect-only', action='store_true', help='Only inspect, do not write')
    args = parser.parse_args()

    if not any([args.datasets, args.pipelines, args.json, args.endpoints, args.schemas, args.data_files, args.all]):
        args.all = True

    if args.all:
        args.datasets = args.pipelines = args.json = args.endpoints = args.schemas = args.data_files = True

    print("=" * 60)
    print("CytoAtlas Documentation Generator")
    print("=" * 60)

    # Dataset documentation
    if args.datasets:
        print("\n## Inspecting Datasets")

        datasets = {
            'cima': {
                'h5ad': Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'),
                'csv': [
                    Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv'),
                    Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv'),
                    Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv'),
                ]
            },
            'inflammation': {
                'h5ad': Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad'),
                'csv': [
                    Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv'),
                ]
            },
            'scatlas': {
                'h5ad': Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'),
                'csv': []
            }
        }

        for atlas, paths in datasets.items():
            print(f"\n### {atlas.upper()}")

            # Inspect H5AD
            print(f"  Inspecting H5AD: {paths['h5ad']}")
            h5ad_meta = inspect_h5ad_metadata(paths['h5ad'])
            print(f"    Cells: {h5ad_meta.get('n_obs', 'N/A'):,}")
            print(f"    Genes: {h5ad_meta.get('n_vars', 'N/A'):,}")
            print(f"    Obs columns: {len(h5ad_meta.get('obs_columns', []))}")

            # Inspect CSVs
            csv_metas = []
            for csv_path in paths['csv']:
                print(f"  Inspecting CSV: {csv_path.name}")
                csv_meta = inspect_csv_metadata(csv_path)
                csv_metas.append(csv_meta)
                print(f"    Columns: {len(csv_meta.get('columns', []))}")

            # Generate doc
            if not args.inspect_only:
                doc = generate_dataset_doc(atlas, h5ad_meta, csv_metas)
                out_path = DOCS_DIR / "datasets" / f"{atlas}.md"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(out_path, 'w') as f:
                    f.write(doc)
                print(f"  Wrote: {out_path}")

    # Pipeline documentation
    if args.pipelines:
        print("\n## Inspecting Pipelines")

        scripts = [
            SCRIPTS_DIR / "00_pilot_analysis.py",
            SCRIPTS_DIR / "01_cima_activity.py",
            SCRIPTS_DIR / "02_inflam_activity.py",
            SCRIPTS_DIR / "03_scatlas_analysis.py",
            SCRIPTS_DIR / "06_preprocess_viz_data.py",
            SCRIPTS_DIR / "07_scatlas_immune_analysis.py",
        ]

        for script_path in scripts:
            if not script_path.exists():
                print(f"  Skipping (not found): {script_path.name}")
                continue

            print(f"\n### {script_path.name}")

            # Extract functions
            functions = extract_functions_from_script(script_path)
            docstring = extract_script_docstring(script_path)

            print(f"  Functions: {len(functions)}")
            for func in functions[:5]:
                print(f"    - {func['name']}() at line {func['lineno']}")
            if len(functions) > 5:
                print(f"    ... and {len(functions) - 5} more")

            # Generate doc
            if not args.inspect_only:
                doc = generate_pipeline_doc(script_path, functions, docstring)

                # Determine output path based on script name
                if 'cima' in script_path.name:
                    out_path = DOCS_DIR / "pipelines" / "cima" / "activity.md"
                elif 'inflam' in script_path.name:
                    out_path = DOCS_DIR / "pipelines" / "inflammation" / "activity.md"
                elif 'scatlas' in script_path.name:
                    if 'immune' in script_path.name:
                        out_path = DOCS_DIR / "pipelines" / "scatlas" / "immune.md"
                    else:
                        out_path = DOCS_DIR / "pipelines" / "scatlas" / "analysis.md"
                elif 'preprocess' in script_path.name:
                    out_path = DOCS_DIR / "pipelines" / "visualization" / "preprocess.md"
                elif 'pilot' in script_path.name:
                    out_path = DOCS_DIR / "pipelines" / "pilot.md"
                else:
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as f:
                    f.write(doc)
                print(f"  Wrote: {out_path}")

    # JSON catalog
    if args.json:
        print("\n## Inspecting JSON Files")

        json_files = []

        for json_path in sorted(VIZ_DATA_DIR.glob("*.json")):
            print(f"  {json_path.name}")
            meta = inspect_json_schema(json_path)
            json_files.append(meta)
            print(f"    Size: {format_bytes(meta.get('size_bytes', 0))}")
            print(f"    Type: {meta.get('type', 'unknown')}")
            print(f"    Records: {meta.get('n_records', '-')}")

        if not args.inspect_only:
            doc = generate_json_catalog(json_files)
            out_path = DOCS_DIR / "outputs" / "visualization" / "index.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, 'w') as f:
                f.write(doc)
            print(f"\nWrote: {out_path}")

    # API endpoint inventory
    if args.endpoints:
        print("\n## Scanning API Endpoints")

        router_dir = PROJECT_ROOT / "cytoatlas-api" / "app" / "routers"
        if router_dir.exists():
            endpoints = extract_api_endpoints(router_dir)
            print(f"  Found {len(endpoints)} routers")
            total_eps = sum(len(eps) for eps in endpoints.values())
            print(f"  Found {total_eps} endpoints")

            if not args.inspect_only:
                doc = generate_endpoint_inventory(endpoints)
                out_path = DOCS_DIR / "API_ENDPOINTS.md"

                with open(out_path, 'w') as f:
                    f.write(doc)
                print(f"  Wrote: {out_path}")
        else:
            print(f"  Skipping (router directory not found): {router_dir}")

    # Pydantic schema inventory
    if args.schemas:
        print("\n## Scanning Pydantic Schemas")

        schemas_dir = PROJECT_ROOT / "cytoatlas-api" / "app" / "schemas"
        if schemas_dir.exists():
            schemas = extract_pydantic_schemas(schemas_dir)
            print(f"  Found {len(schemas)} schema files")
            total_schemas = sum(len(models) for models in schemas.values())
            print(f"  Found {total_schemas} schema models")

            if not args.inspect_only:
                doc = generate_schema_inventory(schemas)
                out_path = DOCS_DIR / "API_SCHEMAS.md"

                with open(out_path, 'w') as f:
                    f.write(doc)
                print(f"  Wrote: {out_path}")
        else:
            print(f"  Skipping (schemas directory not found): {schemas_dir}")

    # Data file inventory
    if args.data_files:
        print("\n## Inventorying Data Files")

        if VIZ_DATA_DIR.exists():
            inventory = inventory_data_files(VIZ_DATA_DIR)
            print(f"  Found {inventory['total_files']} files ({format_bytes(inventory['total_size'])})")

            if not args.inspect_only:
                doc = generate_data_file_inventory(inventory)
                out_path = DOCS_DIR / "DATA_FILES.md"

                with open(out_path, 'w') as f:
                    f.write(doc)
                print(f"  Wrote: {out_path}")
        else:
            print(f"  Skipping (data directory not found): {VIZ_DATA_DIR}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
