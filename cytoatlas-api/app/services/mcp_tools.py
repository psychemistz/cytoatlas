"""MCP-style tool definitions for CytoAtlas chat.

These tools allow the LLM to query CytoAtlas data and create visualizations.
"""

import json
import logging
from typing import Any, Callable

from app.config import get_settings
from app.services.search_service import get_search_service
from app.schemas.search import SearchType

logger = logging.getLogger(__name__)
settings = get_settings()


# Tool definitions (Anthropic format, converted to OpenAI format at bottom)
CYTOATLAS_TOOLS = [
    {
        "name": "search_entity",
        "description": "Search for genes, cytokines, secreted proteins, cell types, diseases, or organs across all CytoAtlas atlases. Use this to find specific entities or explore what's available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'IFNG', 'CD8 T cell', 'liver')"
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["all", "cytokine", "protein", "cell_type", "disease", "organ"],
                    "description": "Filter by entity type (default: all)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_atlas_summary",
        "description": "Get summary statistics for an atlas including number of cells, samples, cell types, and available data types.",
        "input_schema": {
            "type": "object",
            "properties": {
                "atlas_name": {
                    "type": "string",
                    "enum": ["CIMA", "Inflammation", "scAtlas"],
                    "description": "Name of the atlas"
                }
            },
            "required": ["atlas_name"]
        }
    },
    {
        "name": "list_cell_types",
        "description": "List all available cell types in an atlas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "atlas_name": {
                    "type": "string",
                    "enum": ["CIMA", "Inflammation", "scAtlas"],
                    "description": "Name of the atlas"
                }
            },
            "required": ["atlas_name"]
        }
    },
    {
        "name": "list_signatures",
        "description": "List available CytoSig cytokines or SecAct secreted proteins.",
        "input_schema": {
            "type": "object",
            "properties": {
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Type of signatures to list"
                }
            },
            "required": ["signature_type"]
        }
    },
    {
        "name": "get_activity_data",
        "description": "Get cytokine or protein activity values for specific signatures across cell types in an atlas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "atlas_name": {
                    "type": "string",
                    "enum": ["CIMA", "Inflammation", "scAtlas"],
                    "description": "Name of the atlas"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Type of signature"
                },
                "signatures": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of signature names (e.g., ['IFNG', 'TNF'])"
                },
                "cell_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: filter to specific cell types"
                }
            },
            "required": ["atlas_name", "signature_type", "signatures"]
        }
    },
    {
        "name": "get_correlations",
        "description": "Get correlations between signature activity and clinical/biochemistry variables. Only available for CIMA atlas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "signature": {
                    "type": "string",
                    "description": "Signature name (e.g., 'IFNG')"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Type of signature"
                },
                "correlation_type": {
                    "type": "string",
                    "enum": ["age", "bmi", "biochemistry"],
                    "description": "Type of correlation"
                },
                "cell_type": {
                    "type": "string",
                    "description": "Optional: specific cell type"
                }
            },
            "required": ["signature", "signature_type", "correlation_type"]
        }
    },
    {
        "name": "get_disease_activity",
        "description": "Get disease-specific activity differences from the Inflammation Atlas. Compare disease vs healthy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "disease": {
                    "type": "string",
                    "description": "Disease name (e.g., 'COVID-19', 'Rheumatoid Arthritis')"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Type of signature"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top differentially active signatures to return (default: 20)"
                }
            },
            "required": ["disease", "signature_type"]
        }
    },
    {
        "name": "compare_atlases",
        "description": "Compare signature activity patterns across multiple atlases for the same cell types.",
        "input_schema": {
            "type": "object",
            "properties": {
                "signature": {
                    "type": "string",
                    "description": "Signature to compare"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Type of signature"
                },
                "atlases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Atlases to compare (default: all)"
                }
            },
            "required": ["signature", "signature_type"]
        }
    },
    {
        "name": "get_validation_metrics",
        "description": "Get validation metrics for CytoSig/SecAct predictions including marker overlap, biological coherence, and cross-validation scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "atlas_name": {
                    "type": "string",
                    "enum": ["CIMA", "Inflammation", "scAtlas"],
                    "description": "Name of the atlas"
                },
                "validation_type": {
                    "type": "string",
                    "enum": ["marker_overlap", "biological_coherence", "cross_validation", "reproducibility", "all"],
                    "description": "Type of validation to retrieve"
                }
            },
            "required": ["atlas_name"]
        }
    },
    {
        "name": "export_data",
        "description": "Prepare data for download as CSV or JSON. Returns a reference that can be used to generate a download link.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["activity", "correlations", "differential", "comparison"],
                    "description": "Type of data to export"
                },
                "format": {
                    "type": "string",
                    "enum": ["csv", "json"],
                    "description": "Export format"
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters specific to the data type"
                }
            },
            "required": ["data_type", "format", "parameters"]
        }
    },
    {
        "name": "create_visualization",
        "description": "Generate a visualization configuration that will be rendered inline in the chat. Use this to show data visually.",
        "input_schema": {
            "type": "object",
            "properties": {
                "viz_type": {
                    "type": "string",
                    "enum": ["heatmap", "bar_chart", "scatter", "box_plot", "line_chart", "table"],
                    "description": "Type of visualization"
                },
                "title": {
                    "type": "string",
                    "description": "Title for the visualization"
                },
                "data": {
                    "type": "object",
                    "description": "Data for the visualization (format depends on viz_type)"
                },
                "config": {
                    "type": "object",
                    "description": "Additional configuration options"
                }
            },
            "required": ["viz_type", "title", "data"]
        }
    },
    # Documentation MCP Tools
    {
        "name": "get_data_lineage",
        "description": "Trace how any output file was generated. Returns the source script, function, and upstream data files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Name of the output file (e.g., 'cima_correlations.json')"
                }
            },
            "required": ["file_name"]
        }
    },
    {
        "name": "get_column_definition",
        "description": "Get the description and type of a column in a data file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "Name of the file containing the column"
                },
                "column_name": {
                    "type": "string",
                    "description": "Name of the column to describe"
                }
            },
            "required": ["file_name", "column_name"]
        }
    },
    {
        "name": "find_source_script",
        "description": "Find which script generates a specific output file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_file": {
                    "type": "string",
                    "description": "Name or partial name of the output file"
                }
            },
            "required": ["output_file"]
        }
    },
    {
        "name": "list_panel_outputs",
        "description": "List all output files associated with an analysis panel or tab.",
        "input_schema": {
            "type": "object",
            "properties": {
                "panel_name": {
                    "type": "string",
                    "description": "Name of the panel (e.g., 'age-correlation', 'disease-overview', 'organ-signatures')"
                }
            },
            "required": ["panel_name"]
        }
    },
    {
        "name": "get_dataset_info",
        "description": "Get detailed information about a dataset including columns, cell counts, and file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "enum": ["cima", "inflammation", "scatlas", "cytosig", "secact"],
                    "description": "Name of the dataset or signature matrix"
                }
            },
            "required": ["dataset_name"]
        }
    },
]


def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI function-calling format.

    The JSON Schema bodies (input_schema vs parameters) are identical;
    only the wrapping object differs between the two APIs.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in anthropic_tools
    ]


OPENAI_TOOLS = _to_openai_tools(CYTOATLAS_TOOLS)


class ToolExecutor:
    """Executes MCP tools against CytoAtlas data."""

    def __init__(self):
        self.settings = get_settings()
        self._data_cache: dict[str, Any] = {}

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        handler = getattr(self, f"_execute_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return await handler(arguments)
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return {"error": str(e)}

    async def _execute_search_entity(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute search_entity tool."""
        search_service = get_search_service()

        entity_type = SearchType(args.get("entity_type", "all"))
        limit = args.get("limit", 10)

        result = await search_service.search(
            query=args["query"],
            type_filter=entity_type,
            offset=0,
            limit=limit,
        )

        return {
            "total_results": result.total_results,
            "results": [
                {
                    "id": r.id,
                    "name": r.name,
                    "type": r.type.value,
                    "atlases": r.atlases,
                    "description": r.description,
                }
                for r in result.results
            ]
        }

    async def _execute_get_atlas_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_atlas_summary tool."""
        atlas_name = args["atlas_name"]

        # Map to file prefix
        prefix_map = {"CIMA": "cima", "Inflammation": "inflammation", "scAtlas": "scatlas"}
        prefix = prefix_map.get(atlas_name)

        if not prefix:
            return {"error": f"Unknown atlas: {atlas_name}"}

        # Load summary data
        summary_path = self.settings.viz_data_path / f"{prefix}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)

        # Fallback to celltype data for basic stats
        celltype_path = self.settings.viz_data_path / f"{prefix}_celltype.json"
        if celltype_path.exists():
            with open(celltype_path) as f:
                data = json.load(f)
                cell_types = sorted(set(r.get("cell_type") for r in data if r.get("cell_type")))
                cytosig_sigs = sorted(set(
                    r.get("signature") for r in data
                    if r.get("signature_type") == "CytoSig" and r.get("signature")
                ))
                secact_sigs = sorted(set(
                    r.get("signature") for r in data
                    if r.get("signature_type") == "SecAct" and r.get("signature")
                ))
                return {
                    "atlas_name": atlas_name,
                    "n_cell_types": len(cell_types),
                    "n_signatures_cytosig": len(cytosig_sigs),
                    "n_signatures_secact": len(secact_sigs),
                    "cell_types": cell_types[:20],
                }

        return {"error": "Summary data not available"}

    async def _execute_list_cell_types(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute list_cell_types tool."""
        atlas_name = args["atlas_name"]
        prefix_map = {"CIMA": "cima", "Inflammation": "inflammation", "scAtlas": "scatlas"}
        prefix = prefix_map.get(atlas_name)

        if not prefix:
            return {"error": f"Unknown atlas: {atlas_name}"}

        # Use the celltype.json file which has the actual data
        activity_path = self.settings.viz_data_path / f"{prefix}_celltype.json"
        if activity_path.exists():
            with open(activity_path) as f:
                data = json.load(f)
                # Extract unique cell types from the list of records
                cell_types = sorted(set(r.get("cell_type") for r in data if r.get("cell_type")))
                return {
                    "atlas_name": atlas_name,
                    "cell_types": cell_types,
                    "count": len(cell_types)
                }

        return {"error": "Cell type data not available"}

    async def _execute_list_signatures(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute list_signatures tool."""
        sig_type = args["signature_type"]

        # Extract signatures from cima_celltype.json which has all signatures
        celltype_path = self.settings.viz_data_path / "cima_celltype.json"
        if celltype_path.exists():
            with open(celltype_path) as f:
                data = json.load(f)
                # Extract unique signatures of the specified type
                signatures = sorted(set(
                    r.get("signature") for r in data
                    if r.get("signature_type") == sig_type and r.get("signature")
                ))
                return {
                    "signature_type": sig_type,
                    "signatures": signatures,
                    "count": len(signatures)
                }

        return {"error": f"{sig_type} signature data not available"}

    async def _execute_get_activity_data(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_activity_data tool."""
        atlas_name = args["atlas_name"]
        sig_type = args["signature_type"]
        signatures = args["signatures"]
        cell_types_filter = args.get("cell_types")

        prefix_map = {"CIMA": "cima", "Inflammation": "inflammation", "scAtlas": "scatlas"}
        prefix = prefix_map.get(atlas_name)

        if not prefix:
            return {"error": f"Unknown atlas: {atlas_name}"}

        # The actual data files are named {prefix}_celltype.json and contain
        # a list of records with cell_type, signature, signature_type, mean_activity
        activity_path = self.settings.viz_data_path / f"{prefix}_celltype.json"
        if not activity_path.exists():
            return {"error": f"Activity data not available for {atlas_name}"}

        with open(activity_path) as f:
            data = json.load(f)

        # Data is a list of records
        # Filter by signature type and requested signatures
        filtered = [
            r for r in data
            if r.get("signature_type") == sig_type
            and r.get("signature") in signatures
        ]

        if not filtered:
            # Try case-insensitive match
            sig_lower = {s.lower(): s for s in signatures}
            filtered = [
                r for r in data
                if r.get("signature_type") == sig_type
                and r.get("signature", "").lower() in sig_lower
            ]

        if not filtered:
            available_sigs = sorted(set(
                r.get("signature") for r in data
                if r.get("signature_type") == sig_type
            ))[:20]
            return {
                "error": f"Signatures not found: {signatures}",
                "available_signatures": available_sigs
            }

        # Filter by cell types if specified
        if cell_types_filter:
            ct_lower = {ct.lower(): ct for ct in cell_types_filter}
            filtered = [
                r for r in filtered
                if r.get("cell_type", "").lower() in ct_lower
                or r.get("cell_type") in cell_types_filter
            ]

        # Build result organized by signature
        result = {
            "atlas_name": atlas_name,
            "signature_type": sig_type,
            "signatures": list(set(r.get("signature") for r in filtered)),
            "cell_types": sorted(set(r.get("cell_type") for r in filtered)),
            "activity": {}
        }

        for record in filtered:
            sig_name = record.get("signature")
            ct_name = record.get("cell_type")
            activity = record.get("mean_activity")

            if sig_name not in result["activity"]:
                result["activity"][sig_name] = {}
            result["activity"][sig_name][ct_name] = activity

        # Sort cell types by activity for the first signature
        if result["activity"]:
            first_sig = list(result["activity"].keys())[0]
            sorted_cts = sorted(
                result["activity"][first_sig].items(),
                key=lambda x: x[1] if x[1] is not None else 0,
                reverse=True
            )
            result["cell_types_ranked"] = [
                {"cell_type": ct, "activity": act}
                for ct, act in sorted_cts
            ]

        return result

    async def _execute_get_correlations(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_correlations tool."""
        signature = args["signature"]
        sig_type = args["signature_type"]
        corr_type = args["correlation_type"]
        cell_type = args.get("cell_type")

        sig_prefix = "cytosig" if sig_type == "CytoSig" else "secact"
        corr_path = self.settings.viz_data_path / f"cima_{sig_prefix}_{corr_type}_correlations.json"

        if not corr_path.exists():
            return {"error": f"Correlation data not available for {corr_type}"}

        with open(corr_path) as f:
            data = json.load(f)

        if signature not in data:
            return {"error": f"Signature {signature} not found in correlation data"}

        correlations = data[signature]

        if cell_type:
            correlations = [c for c in correlations if c.get("cell_type") == cell_type]

        return {
            "signature": signature,
            "signature_type": sig_type,
            "correlation_type": corr_type,
            "correlations": correlations[:50]  # Limit for readability
        }

    async def _execute_get_disease_activity(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_disease_activity tool."""
        disease = args["disease"]
        sig_type = args["signature_type"]
        top_n = args.get("top_n", 20)

        sig_prefix = "cytosig" if sig_type == "CytoSig" else "secact"
        diff_path = self.settings.viz_data_path / f"inflam_{sig_prefix}_differential.json"

        if not diff_path.exists():
            return {"error": "Differential activity data not available"}

        with open(diff_path) as f:
            data = json.load(f)

        # Find disease in data
        disease_data = None
        for d in data.get("diseases", []):
            if d.lower() == disease.lower() or disease.lower() in d.lower():
                disease_data = data.get(d)
                break

        if not disease_data:
            return {
                "error": f"Disease not found: {disease}",
                "available_diseases": data.get("diseases", [])[:20]
            }

        # Sort by effect size and take top N
        sorted_data = sorted(
            disease_data,
            key=lambda x: abs(x.get("effect_size", 0)),
            reverse=True
        )[:top_n]

        return {
            "disease": disease,
            "signature_type": sig_type,
            "differential_signatures": sorted_data
        }

    async def _execute_compare_atlases(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute compare_atlases tool."""
        signature = args["signature"]
        sig_type = args["signature_type"]
        atlases = args.get("atlases", ["CIMA", "Inflammation", "scAtlas"])

        prefix_map = {"CIMA": "cima", "Inflammation": "inflammation", "scAtlas": "scatlas"}

        result = {
            "signature": signature,
            "signature_type": sig_type,
            "comparison": {}
        }

        for atlas_name in atlases:
            prefix = prefix_map.get(atlas_name)
            if not prefix:
                continue

            celltype_path = self.settings.viz_data_path / f"{prefix}_celltype.json"
            if not celltype_path.exists():
                continue

            with open(celltype_path) as f:
                data = json.load(f)

            # Filter for the signature
            sig_lower = signature.lower()
            filtered = [
                r for r in data
                if r.get("signature_type") == sig_type
                and (r.get("signature") == signature or r.get("signature", "").lower() == sig_lower)
            ]

            if not filtered:
                continue

            atlas_data = {}
            for record in filtered:
                ct_name = record.get("cell_type")
                activity = record.get("mean_activity")
                if ct_name and activity is not None:
                    atlas_data[ct_name] = activity

            result["comparison"][atlas_name] = atlas_data

        return result

    async def _execute_get_validation_metrics(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_validation_metrics tool."""
        atlas_name = args["atlas_name"]
        validation_type = args.get("validation_type", "all")

        prefix_map = {"CIMA": "cima", "Inflammation": "inflam", "scAtlas": "scatlas"}
        prefix = prefix_map.get(atlas_name)

        if not prefix:
            return {"error": f"Unknown atlas: {atlas_name}"}

        validation_path = self.settings.viz_data_path / f"{prefix}_validation.json"
        if not validation_path.exists():
            return {"error": "Validation data not available"}

        with open(validation_path) as f:
            data = json.load(f)

        if validation_type == "all":
            return data

        if validation_type in data:
            return {validation_type: data[validation_type]}

        return {"error": f"Validation type not found: {validation_type}"}

    async def _execute_export_data(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute export_data tool."""
        data_type = args["data_type"]
        format = args["format"]
        parameters = args.get("parameters", {})

        # Generate a unique export ID
        import uuid
        export_id = str(uuid.uuid4())[:8]

        # Store parameters for later download
        self._data_cache[export_id] = {
            "data_type": data_type,
            "format": format,
            "parameters": parameters,
        }

        return {
            "export_id": export_id,
            "download_available": True,
            "format": format,
            "description": f"Export {data_type} data in {format} format"
        }

    async def _execute_create_visualization(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute create_visualization tool."""
        viz_type = args["viz_type"]
        title = args["title"]
        data = args["data"]
        config = args.get("config", {})

        # Validate data structure based on viz type
        if viz_type == "heatmap":
            required = ["x_labels", "y_labels", "values"]
        elif viz_type == "bar_chart":
            required = ["labels", "values"]
        elif viz_type == "scatter":
            required = ["x", "y"]
        elif viz_type == "box_plot":
            required = ["labels", "values"]
        elif viz_type == "table":
            required = ["headers", "rows"]
        else:
            required = []

        missing = [r for r in required if r not in data]
        if missing:
            return {"error": f"Missing required data fields for {viz_type}: {missing}"}

        import uuid
        container_id = f"viz-{uuid.uuid4().hex[:8]}"

        return {
            "visualization": {
                "type": viz_type,
                "title": title,
                "data": data,
                "config": config,
                "container_id": container_id,
            }
        }

    async def _execute_get_data_lineage(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_data_lineage tool - trace file generation."""
        file_name = args["file_name"]

        registry = self._load_registry()
        if not registry:
            return {"error": "Documentation registry not available"}

        files = registry.get("files", {})

        # Find file in registry
        if file_name in files:
            file_info = files[file_name]
            return {
                "file": file_name,
                "source_script": file_info.get("source_script"),
                "source_function": file_info.get("source_function"),
                "upstream": file_info.get("upstream", []),
                "description": file_info.get("description"),
                "atlas": file_info.get("atlas"),
            }

        # Try partial match
        matches = [f for f in files if file_name.lower() in f.lower()]
        if matches:
            return {
                "error": f"Exact match not found for '{file_name}'",
                "similar_files": matches[:5]
            }

        return {"error": f"File not found: {file_name}"}

    async def _execute_get_column_definition(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_column_definition tool."""
        file_name = args["file_name"]
        column_name = args["column_name"]

        registry = self._load_registry()
        if not registry:
            return {"error": "Documentation registry not available"}

        files = registry.get("files", {})

        if file_name in files:
            schema = files[file_name].get("schema", {})
            if column_name in schema:
                return {
                    "file": file_name,
                    "column": column_name,
                    "type": schema[column_name],
                }

        # Common column definitions
        common_columns = {
            "cell_type": {"type": "string", "description": "Cell type annotation (e.g., CD4+ T, Monocyte)"},
            "signature": {"type": "string", "description": "Cytokine or protein name (e.g., IL6, TGFB1)"},
            "signature_type": {"type": "string", "description": "CytoSig (44 cytokines) or SecAct (1,249 proteins)"},
            "rho": {"type": "number", "description": "Spearman correlation coefficient (-1 to 1)"},
            "pvalue": {"type": "number", "description": "Statistical p-value (0 to 1)"},
            "fdr": {"type": "number", "description": "FDR-corrected p-value (Benjamini-Hochberg)"},
            "activity_diff": {"type": "number", "description": "Activity difference (group1_mean - group2_mean, NOT log2FC)"},
            "mean_activity": {"type": "number", "description": "Mean activity z-score (typically -3 to +3)"},
            "n_cells": {"type": "integer", "description": "Number of cells in the group"},
            "organ": {"type": "string", "description": "Organ/tissue name (scAtlas)"},
            "disease": {"type": "string", "description": "Disease diagnosis (Inflammation Atlas)"},
            "comparison": {"type": "string", "description": "Comparison label (e.g., 'sex_Male_vs_Female')"},
            "neg_log10_pval": {"type": "number", "description": "-log10(pvalue) for volcano plots"},
        }

        if column_name in common_columns:
            return {
                "column": column_name,
                **common_columns[column_name]
            }

        return {"error": f"Column definition not found: {column_name}"}

    async def _execute_find_source_script(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute find_source_script tool."""
        output_file = args["output_file"].lower()

        registry = self._load_registry()
        if not registry:
            return {"error": "Documentation registry not available"}

        files = registry.get("files", {})
        scripts = registry.get("scripts", {})

        # Search in files
        matches = []
        for fname, finfo in files.items():
            if output_file in fname.lower():
                matches.append({
                    "file": fname,
                    "source_script": finfo.get("source_script"),
                    "source_function": finfo.get("source_function"),
                })

        if matches:
            return {"matches": matches}

        # Search in scripts output dirs
        script_matches = []
        for script_name, script_info in scripts.items():
            output_dir = script_info.get("output_dir", "")
            if output_file in output_dir.lower() or output_file in script_name.lower():
                script_matches.append({
                    "script": script_name,
                    "output_dir": output_dir,
                    "description": script_info.get("description"),
                })

        if script_matches:
            return {"script_matches": script_matches}

        return {"error": f"No source script found for: {output_file}"}

    async def _execute_list_panel_outputs(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute list_panel_outputs tool."""
        panel_name = args["panel_name"].lower().replace(" ", "-")

        registry = self._load_registry()
        if not registry:
            return {"error": "Documentation registry not available"}

        panels = registry.get("panels", {})
        files = registry.get("files", {})

        # Find panel
        panel_info = None
        for pname, pinfo in panels.items():
            if panel_name in pname.lower():
                panel_info = pinfo
                panel_name = pname
                break

        if panel_info:
            json_files = panel_info.get("json_files", [])
            file_details = []
            for jf in json_files:
                if jf in files:
                    file_details.append({
                        "file": jf,
                        "description": files[jf].get("description"),
                        "api_endpoints": files[jf].get("api_endpoints", []),
                    })

            return {
                "panel": panel_name,
                "tab": panel_info.get("tab"),
                "title": panel_info.get("title"),
                "chart_type": panel_info.get("chart_type"),
                "json_files": file_details,
            }

        # Search files by panel name
        matching_files = []
        for fname, finfo in files.items():
            if panel_name in str(finfo.get("panels", [])).lower():
                matching_files.append({
                    "file": fname,
                    "panels": finfo.get("panels", []),
                    "description": finfo.get("description"),
                })

        if matching_files:
            return {"matching_files": matching_files}

        available_panels = list(panels.keys())
        return {
            "error": f"Panel not found: {panel_name}",
            "available_panels": available_panels
        }

    async def _execute_get_dataset_info(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_dataset_info tool."""
        dataset_name = args["dataset_name"].lower()

        registry = self._load_registry()
        if not registry:
            return {"error": "Documentation registry not available"}

        # Check atlases
        atlases = registry.get("atlases", {})
        if dataset_name in atlases:
            return atlases[dataset_name]

        # Check signatures
        signatures = registry.get("signatures", {})
        if dataset_name in signatures:
            return signatures[dataset_name]

        return {
            "error": f"Dataset not found: {dataset_name}",
            "available_datasets": list(atlases.keys()) + list(signatures.keys())
        }

    def _load_registry(self) -> dict[str, Any]:
        """Load the documentation registry."""
        from pathlib import Path

        registry_path = Path("/vf/users/parks34/projects/2secactpy/docs/registry.json")
        if not registry_path.exists():
            return {}

        try:
            with open(registry_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}


# Singleton instance
_tool_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    """Get or create the tool executor singleton."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor
