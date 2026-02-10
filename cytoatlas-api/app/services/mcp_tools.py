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
                    "enum": ["all", "cytokine", "protein", "cell_type", "disease", "organ", "drug", "perturbation"],
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
                    "enum": ["cima", "inflammation", "scatlas", "parse10m", "tahoe", "spatial_corpus", "cytosig", "secact"],
                    "description": "Name of the dataset or signature matrix"
                }
            },
            "required": ["dataset_name"]
        }
    },
    {
        "name": "query_perturbation_data",
        "description": "Query cytokine perturbation (parse_10M) or drug perturbation (Tahoe) data. Get activity changes after treatment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "enum": ["parse10m", "tahoe"],
                    "description": "Which perturbation dataset to query"
                },
                "treatment": {
                    "type": "string",
                    "description": "Cytokine name (parse_10M) or drug name (Tahoe)"
                },
                "cell_type_or_line": {
                    "type": "string",
                    "description": "PBMC cell type (parse_10M) or cancer cell line (Tahoe)"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Signature matrix type (default: CytoSig)"
                }
            },
            "required": ["dataset"]
        }
    },
    {
        "name": "get_ground_truth_validation",
        "description": "Get ground truth validation results comparing CytoSig/SecAct predicted activity against actual cytokine treatment response in parse_10M data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Signature matrix type"
                },
                "cytokine": {
                    "type": "string",
                    "description": "Specific cytokine to validate (optional)"
                },
                "cell_type": {
                    "type": "string",
                    "description": "PBMC cell type (optional)"
                }
            }
        }
    },
    {
        "name": "get_drug_sensitivity",
        "description": "Get drug sensitivity data from Tahoe-100M. Returns activity changes across cancer cell lines for specific drugs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drug": {
                    "type": "string",
                    "description": "Drug name"
                },
                "cell_line": {
                    "type": "string",
                    "description": "Cancer cell line name (optional)"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Signature matrix type"
                }
            },
            "required": ["drug"]
        }
    },
    {
        "name": "get_dose_response",
        "description": "Get dose-response data from Tahoe Plate 13 (3 dose levels for 25 drugs across 50 cell lines).",
        "input_schema": {
            "type": "object",
            "properties": {
                "drug": {
                    "type": "string",
                    "description": "Drug name"
                },
                "cell_line": {
                    "type": "string",
                    "description": "Cancer cell line (optional)"
                },
                "signature": {
                    "type": "string",
                    "description": "Specific cytokine/protein signature (optional)"
                }
            },
            "required": ["drug"]
        }
    },
    {
        "name": "get_spatial_activity",
        "description": "Get spatial transcriptomics activity data from SpatialCorpus-110M. Query by technology, tissue, or signature.",
        "input_schema": {
            "type": "object",
            "properties": {
                "technology": {
                    "type": "string",
                    "enum": ["Visium", "Xenium", "MERFISH", "MERSCOPE", "CosMx"],
                    "description": "Spatial technology platform"
                },
                "tissue": {
                    "type": "string",
                    "description": "Tissue/organ type"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Signature matrix type"
                }
            }
        }
    },
    {
        "name": "compare_technologies",
        "description": "Compare activity patterns across spatial transcriptomics technologies for the same tissue types.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tissue": {
                    "type": "string",
                    "description": "Tissue to compare across technologies"
                },
                "signature_type": {
                    "type": "string",
                    "enum": ["CytoSig", "SecAct"],
                    "description": "Signature matrix type"
                }
            },
            "required": ["tissue"]
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

    # Tools whose results can be auto-visualized
    _DATA_TOOLS = {
        "get_activity_data", "compare_atlases", "get_disease_activity",
        "get_correlations", "get_validation_metrics",
    }

    def __init__(self):
        self.settings = get_settings()
        self._data_cache: dict[str, Any] = {}
        # Cache of recent data tool results for auto-visualization fallback
        self._recent_data_results: list[dict[str, Any]] = []

    def clear_recent_data(self):
        """Clear cached data results. Call at start of each conversation turn."""
        self._recent_data_results.clear()

    # Common parameter name mistakes from Mistral (wrong → correct)
    _PARAM_ALIASES = {
        "atlases": "atlas_name",      # Mistral confuses with compare_atlases
        "atlas": "atlas_name",
        "sig_type": "signature_type",
        "type": "signature_type",
        "sigs": "signatures",
        "signature_names": "signatures",
        "query_text": "query",
        "cell_type_filter": "cell_types",
    }

    def _normalize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize parameter names to handle LLM mistakes.

        Maps common parameter name variations to the expected names.
        Also handles cases where atlas_name is provided as a list.
        """
        normalized = {}
        for key, value in args.items():
            canonical = self._PARAM_ALIASES.get(key, key)
            normalized[canonical] = value

        # If atlas_name was provided as a list (from atlases), extract first value
        if isinstance(normalized.get("atlas_name"), list):
            normalized["atlas_name"] = normalized["atlas_name"][0]

        return normalized

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        handler = getattr(self, f"_execute_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        # Normalize parameter names to handle LLM mistakes
        arguments = self._normalize_args(arguments)

        try:
            result = await handler(arguments)
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return {"error": str(e)}

        # Cache successful data tool results for auto-visualization fallback
        if tool_name in self._DATA_TOOLS and "error" not in result:
            self._recent_data_results.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            })
            # Keep only last 5 results
            if len(self._recent_data_results) > 5:
                self._recent_data_results = self._recent_data_results[-5:]

        return result

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
        records = self._load_celltype_records(prefix)
        if records is not None:
            data = records
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

        data = self._load_celltype_records(prefix)
        if data is not None:
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

        # Try each atlas to find signatures
        for prefix in ["cima", "inflammation", "scatlas"]:
            data = self._load_celltype_records(prefix)
            if data is not None:
                signatures = sorted(set(
                    r.get("signature") for r in data
                    if r.get("signature_type") == sig_type and r.get("signature")
                ))
                if signatures:
                    return {
                        "signature_type": sig_type,
                        "signatures": signatures,
                        "count": len(signatures),
                    }

        return {"error": f"{sig_type} signature data not available"}

    def _load_celltype_records(self, prefix: str) -> list[dict] | None:
        """Load cell type activity records, handling different file formats.

        Files may be:
        - {prefix}_celltype.json (flat list of records)
        - {prefix}_celltypes.json (nested dict with 'data' key)
        """
        # Try flat list format first
        for fname in [f"{prefix}_celltype.json", f"{prefix}_celltypes.json"]:
            path = self.settings.viz_data_path / fname
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                # Handle nested format (scAtlas)
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                if isinstance(data, list):
                    return data
        return None

    async def _execute_get_activity_data(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_activity_data tool."""
        atlas_name = args["atlas_name"]
        sig_type = args["signature_type"]
        signatures = args.get("signatures", [])
        cell_types_filter = args.get("cell_types")

        prefix_map = {"CIMA": "cima", "Inflammation": "inflammation", "scAtlas": "scatlas"}
        prefix = prefix_map.get(atlas_name)

        if not prefix:
            return {"error": f"Unknown atlas: {atlas_name}"}

        data = self._load_celltype_records(prefix)
        if data is None:
            return {"error": f"Activity data not available for {atlas_name}"}

        # Handle wildcard/empty signatures: return top signatures for the cell type
        # This handles cases like signatures=["SecAct"], signatures=["all"], or signatures=[]
        is_wildcard = (
            not signatures
            or signatures == [sig_type]
            or signatures == ["all"]
            or signatures == ["*"]
        )

        if is_wildcard and cell_types_filter:
            # Common cell type aliases for flexible matching
            _CT_ALIASES = {
                "tams": "macrophage",
                "tumor_associated_macrophages": "macrophage",
                "tumor_associated_macrophage": "macrophage",
                "tam": "macrophage",
                "tregs": "regulatory",
                "treg": "regulatory",
                "nk_cells": "nk",
                "b_cells": "b_",
                "t_cells": "t_",
            }

            # Build match terms from cell_types_filter
            ct_lower_map = {ct.lower().replace(" ", "_").replace("-", "_"): ct for ct in cell_types_filter}
            # Expand aliases
            expanded_terms = set(ct_lower_map.keys())
            for term in list(expanded_terms):
                alias = _CT_ALIASES.get(term)
                if alias:
                    expanded_terms.add(alias)

            # Return top N most active signatures for the specified cell types
            ct_records = [
                r for r in data
                if r.get("signature_type") == sig_type
                and any(
                    term in r.get("cell_type", "").lower().replace(" ", "_")
                    for term in expanded_terms
                )
            ]
            if ct_records:
                # Sort by absolute activity
                ct_records.sort(key=lambda r: abs(r.get("mean_activity", 0)), reverse=True)
                top_records = ct_records[:25]
                result = {
                    "atlas_name": atlas_name,
                    "signature_type": sig_type,
                    "cell_types": sorted(set(r.get("cell_type") for r in top_records)),
                    "query_mode": "top_signatures",
                    "signatures": sorted(set(r.get("signature") for r in top_records)),
                    "cell_types_ranked": [
                        {"cell_type": r.get("cell_type"), "signature": r.get("signature"),
                         "activity": r.get("mean_activity")}
                        for r in top_records
                    ],
                }
                return result
            return {
                "error": f"No {sig_type} data for cell types: {cell_types_filter}",
                "available_cell_types": sorted(set(r.get("cell_type") for r in data if r.get("signature_type") == sig_type))[:30],
            }

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

        # For scAtlas: add organ-level summary (average activity per organ)
        if atlas_name == "scAtlas" and any(r.get("organ") for r in filtered):
            from collections import defaultdict
            organ_activities: dict[str, list[float]] = defaultdict(list)
            for record in filtered:
                organ = record.get("organ")
                activity = record.get("mean_activity")
                if organ and activity is not None:
                    organ_activities[organ].append(activity)

            organ_summary = sorted(
                [
                    {"organ": organ, "mean_activity": round(sum(acts) / len(acts), 4), "n_cell_types": len(acts)}
                    for organ, acts in organ_activities.items()
                ],
                key=lambda x: x["mean_activity"],
                reverse=True,
            )
            result["organ_summary"] = organ_summary

            # For large scAtlas results, remove per-cell-type detail to avoid truncation
            if len(result.get("cell_types", [])) > 40:
                result.pop("activity", None)
                result.pop("cell_types", None)
                ranked = result.get("cell_types_ranked", [])
                if len(ranked) > 40:
                    result["cell_types_ranked"] = ranked[:20] + ranked[-20:]
                    result["_cell_types_truncated"] = True

        return result

    async def _execute_get_correlations(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_correlations tool."""
        signature = args["signature"]
        sig_type = args["signature_type"]
        corr_type = args["correlation_type"]
        cell_type = args.get("cell_type")

        # Main correlations file: cima_correlations.json with age/bmi/biochemistry keys
        corr_path = self.settings.viz_data_path / "cima_correlations.json"

        if not corr_path.exists():
            return {"error": "Correlation data not available"}

        with open(corr_path) as f:
            data = json.load(f)

        if corr_type not in data:
            return {
                "error": f"Correlation type '{corr_type}' not found",
                "available_types": list(data.keys()),
            }

        records = data[corr_type]

        # Filter by signature name (field is 'protein') and signature type (field is 'signature')
        sig_type_map = {"CytoSig": "CytoSig", "SecAct": "SecAct"}
        correlations = [
            r for r in records
            if r.get("protein") == signature
            and r.get("signature") == sig_type_map.get(sig_type, sig_type)
        ]

        # Try case-insensitive match
        if not correlations:
            sig_lower = signature.lower()
            correlations = [
                r for r in records
                if r.get("protein", "").lower() == sig_lower
                and r.get("signature") == sig_type_map.get(sig_type, sig_type)
            ]

        if not correlations:
            available = sorted(set(
                r.get("protein") for r in records
                if r.get("signature") == sig_type_map.get(sig_type, sig_type)
            ))[:20]
            return {
                "error": f"Signature '{signature}' not found in {corr_type} correlations",
                "available_signatures": available,
            }

        # Try cell-type-specific correlations (always when available for richer data)
        ct_corr_path = self.settings.viz_data_path / "cima_celltype_correlations.json"
        if ct_corr_path.exists():
            with open(ct_corr_path) as f:
                ct_data = json.load(f)

            # File is a dict with {age: [...], bmi: [...]} structure
            ct_records = ct_data.get(corr_type, [])
            if ct_records:
                sig_matched = sig_type_map.get(sig_type, sig_type)

                # Get all cell types for this signature first
                all_ct_correlations = [
                    r for r in ct_records
                    if r.get("protein") == signature
                    and r.get("signature") == sig_matched
                ]

                if cell_type and all_ct_correlations:
                    # Flexible cell type matching: try exact, contains, prefix
                    ct_lower = cell_type.lower().replace(" ", "_").replace("-", "_")
                    # Extract key prefix (e.g., "cd4" from "cd4_t_cell")
                    ct_prefix = ct_lower.split("_")[0]

                    ct_correlations = [
                        r for r in all_ct_correlations
                        if r.get("cell_type", "").lower() == ct_lower
                        or ct_lower in r.get("cell_type", "").lower()
                        or r.get("cell_type", "").lower().startswith(ct_prefix + "_")
                        or r.get("cell_type", "").lower() == ct_prefix
                    ]
                    if ct_correlations:
                        correlations = ct_correlations
                elif all_ct_correlations:
                    correlations = all_ct_correlations

        return {
            "signature": signature,
            "signature_type": sig_type,
            "correlation_type": corr_type,
            "correlations": correlations[:50],
        }

    async def _execute_get_disease_activity(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute get_disease_activity tool."""
        disease = args["disease"]
        sig_type = args["signature_type"]
        top_n = args.get("top_n", 20)

        diff_path = self.settings.viz_data_path / "inflammation_differential.json"

        if not diff_path.exists():
            return {"error": "Differential activity data not available"}

        with open(diff_path) as f:
            data = json.load(f)

        # Data is a flat list of records with 'disease', 'protein', 'signature' (=sig type) fields
        all_diseases = sorted(set(r.get("disease", "") for r in data if r.get("disease")))

        # Find matching disease (case-insensitive, partial match)
        matched_disease = None
        for d in all_diseases:
            if d.lower() == disease.lower() or disease.lower() in d.lower():
                matched_disease = d
                break

        if not matched_disease:
            return {
                "error": f"Disease not found: {disease}",
                "available_diseases": all_diseases
            }

        # Filter records for this disease and signature type
        # Note: field 'signature' in this file means CytoSig/SecAct type
        sig_type_map = {"CytoSig": "CytoSig", "SecAct": "SecAct"}
        disease_records = [
            r for r in data
            if r.get("disease") == matched_disease
            and r.get("signature") == sig_type_map.get(sig_type, sig_type)
        ]

        if not disease_records:
            return {
                "error": f"No {sig_type} data for {matched_disease}",
                "available_diseases": all_diseases,
            }

        # Sort by absolute activity difference and take top N
        sorted_data = sorted(
            disease_records,
            key=lambda x: abs(x.get("activity_diff", 0)),
            reverse=True
        )[:top_n]

        return {
            "disease": matched_disease,
            "signature_type": sig_type,
            "n_total": len(disease_records),
            "differential_signatures": [
                {
                    "signature": r.get("protein"),
                    "activity_diff": r.get("activity_diff"),
                    "pvalue": r.get("pvalue"),
                    "qvalue": r.get("qvalue"),
                    "mean_disease": r.get("mean_g1"),
                    "mean_healthy": r.get("mean_g2"),
                }
                for r in sorted_data
            ],
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

            data = self._load_celltype_records(prefix)
            if data is None:
                continue

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

        prefix_map = {"CIMA": "cima", "Inflammation": "inflammation", "scAtlas": "scatlas"}
        prefix = prefix_map.get(atlas_name)

        if not prefix:
            return {"error": f"Unknown atlas: {atlas_name}"}

        # Check validation/ subdirectory first, then root
        for vpath in [
            self.settings.viz_data_path / "validation" / f"{prefix}_validation.json",
            self.settings.viz_data_path / f"{prefix}_validation.json",
            self.settings.viz_data_path / f"{prefix}_atlas_validation.json",
        ]:
            if vpath.exists():
                with open(vpath) as f:
                    data = json.load(f)

                if validation_type == "all":
                    # Summarize to avoid huge response
                    summary = {
                        "atlas": data.get("atlas", atlas_name),
                        "signature_types": data.get("signature_types", []),
                        "available_metrics": list(data.keys()),
                    }
                    for key in ["gene_coverage", "cv_stability", "biological_associations"]:
                        if key in data:
                            items = data[key]
                            if isinstance(items, list):
                                summary[key] = {
                                    "count": len(items),
                                    "sample": items[:5] if items else [],
                                }
                            else:
                                summary[key] = items
                    return summary

                if validation_type in data:
                    items = data[validation_type]
                    if isinstance(items, list) and len(items) > 30:
                        return {
                            validation_type: items[:30],
                            "_truncated": True,
                            "_total": len(items),
                        }
                    return {validation_type: items}

                return {
                    "error": f"Validation type '{validation_type}' not found",
                    "available_types": list(data.keys()),
                }

        return {"error": f"Validation data not available for {atlas_name}"}

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
        """Execute create_visualization tool.

        If args are empty/incomplete (garbled JSON from Mistral), auto-generates
        a visualization from the most recent cached data tool result.
        """
        viz_type = args.get("viz_type")
        title = args.get("title")
        data = args.get("data")
        config = args.get("config", {})

        # If args are empty or missing required fields, auto-generate from cache
        if not viz_type or not data:
            auto = self._auto_visualize()
            if auto:
                logger.info("Auto-generated visualization from cached data (garbled args fallback)")
                return auto
            return {"error": "create_visualization requires viz_type, title, and data"}

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
            # Try auto-generation as fallback
            auto = self._auto_visualize()
            if auto:
                logger.info("Auto-generated visualization (missing fields: %s)", missing)
                return auto
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

    def _auto_visualize(self) -> dict[str, Any] | None:
        """Auto-generate a visualization from the most recent cached data result.

        Returns a visualization dict or None if no suitable data is cached.
        """
        if not self._recent_data_results:
            return None

        import uuid

        # Pop the oldest unused cached result
        cached = self._recent_data_results.pop(0)
        tool_name = cached["tool_name"]
        args = cached["arguments"]
        result = cached["result"]

        # Handle pending visualization from compare_atlases (second chart)
        if tool_name == "_pending_viz" and "_viz" in result:
            return {"visualization": result["_viz"]}

        if tool_name == "get_activity_data":
            return self._auto_viz_activity(result, args)
        elif tool_name == "compare_atlases":
            return self._auto_viz_comparison(result, args)
        elif tool_name == "get_disease_activity":
            return self._auto_viz_disease(result, args)
        elif tool_name == "get_correlations":
            return self._auto_viz_correlations(result, args)
        elif tool_name == "get_validation_metrics":
            return self._auto_viz_validation(result, args)

        return None

    def _auto_viz_activity(self, result: dict, args: dict) -> dict[str, Any] | None:
        """Generate bar chart from get_activity_data result."""
        import uuid

        ranked = result.get("cell_types_ranked")
        if not ranked:
            # Try organ_summary for scAtlas
            organ_summary = result.get("organ_summary")
            if organ_summary:
                labels = [o["organ"] for o in organ_summary[:25]]
                values = [o["mean_activity"] for o in organ_summary[:25]]
                sigs = result.get("signatures", [""])
                title = f"{sigs[0]} Activity Across Organs ({args.get('atlas_name', 'scAtlas')})"
                return {
                    "visualization": {
                        "type": "bar_chart",
                        "title": title,
                        "data": {"labels": labels, "values": values},
                        "config": {"x_label": "Organ", "y_label": "Mean Activity (z-score)"},
                        "container_id": f"viz-{uuid.uuid4().hex[:8]}",
                    }
                }
            return None

        # Handle top_signatures mode (wildcard query for a cell type)
        if result.get("query_mode") == "top_signatures":
            labels = [r.get("signature", "?") for r in ranked]
            values = [r.get("activity", 0) for r in ranked]
            cell_types = result.get("cell_types", [""])
            atlas = args.get("atlas_name", "")
            title = f"Top {result.get('signature_type', '')} Signatures in {', '.join(cell_types)} ({atlas})"
        else:
            labels = [r["cell_type"] for r in ranked]
            values = [r["activity"] for r in ranked]
            sigs = result.get("signatures", [""])
            atlas = args.get("atlas_name", "")
            title = f"{sigs[0]} Activity Across Cell Types ({atlas})"

        return {
            "visualization": {
                "type": "bar_chart",
                "title": title,
                "data": {"labels": labels, "values": values},
                "config": {"x_label": "Signature" if result.get("query_mode") == "top_signatures" else "Cell Type",
                           "y_label": "Activity (z-score)"},
                "container_id": f"viz-{uuid.uuid4().hex[:8]}",
            }
        }

    def _auto_viz_comparison(self, result: dict, args: dict) -> dict[str, Any] | None:
        """Generate bar charts from compare_atlases result (one per atlas)."""
        import uuid

        comparison = result.get("comparison", {})
        if not comparison:
            return None

        sig = result.get("signature", "")
        visualizations = []

        for atlas_name, atlas_data in comparison.items():
            if not atlas_data:
                continue
            sorted_items = sorted(atlas_data.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_items]
            values = [round(item[1], 4) for item in sorted_items]

            visualizations.append({
                "type": "bar_chart",
                "title": f"{sig} Activity in {atlas_name}",
                "data": {"labels": labels, "values": values},
                "config": {"x_label": "Cell Type", "y_label": "Activity (z-score)"},
                "container_id": f"viz-{uuid.uuid4().hex[:8]}",
            })

        if not visualizations:
            return None

        # Return first visualization; if there are more, push them back into cache
        # as additional visualization results
        first_viz = {"visualization": visualizations[0]}
        if len(visualizations) > 1:
            # Store remaining as pending auto-viz results
            for extra_viz in visualizations[1:]:
                self._recent_data_results.insert(0, {
                    "tool_name": "_pending_viz",
                    "arguments": {},
                    "result": {"_viz": extra_viz},
                })
        return first_viz

    def _auto_viz_disease(self, result: dict, args: dict) -> dict[str, Any] | None:
        """Generate bar chart from get_disease_activity result."""
        import uuid

        sigs = result.get("differential_signatures", [])
        if not sigs:
            return None

        labels = [s.get("protein", s.get("signature", "?")) for s in sigs[:25]]
        values = [s.get("activity_diff", 0) for s in sigs[:25]]
        disease = args.get("disease", "")
        title = f"Top Differentially Active Cytokines in {disease}"

        return {
            "visualization": {
                "type": "bar_chart",
                "title": title,
                "data": {"labels": labels, "values": values},
                "config": {"x_label": "Cytokine", "y_label": "Activity Difference (z-score)"},
                "container_id": f"viz-{uuid.uuid4().hex[:8]}",
            }
        }

    def _auto_viz_correlations(self, result: dict, args: dict) -> dict[str, Any] | None:
        """Generate bar chart from get_correlations result."""
        import uuid

        correlations = result.get("correlations", [])
        if not correlations:
            return None

        sig = args.get("signature", "")
        corr_type = args.get("correlation_type", "")

        # Sort by absolute rho value (most significant first)
        sorted_corrs = sorted(correlations, key=lambda c: abs(c.get("rho", 0)), reverse=True)

        labels = []
        values = []
        for c in sorted_corrs[:25]:
            label = c.get("cell_type", c.get("feature", c.get("protein", "?")))
            labels.append(label)
            values.append(round(c.get("rho", c.get("correlation", 0)), 4))

        # Even a single record is worth showing
        title = f"{sig} {corr_type.title()} Correlation Across Cell Types"

        return {
            "visualization": {
                "type": "bar_chart",
                "title": title,
                "data": {"labels": labels, "values": values},
                "config": {"x_label": "Cell Type", "y_label": f"Correlation with {corr_type.title()} (rho)"},
                "container_id": f"viz-{uuid.uuid4().hex[:8]}",
            }
        }

    def _auto_viz_validation(self, result: dict, args: dict) -> dict[str, Any] | None:
        """Generate bar chart from get_validation_metrics result."""
        import uuid

        atlas = args.get("atlas_name", "")

        # Extract gene_coverage as the most visualizable metric
        gene_cov = result.get("gene_coverage", {})
        if isinstance(gene_cov, dict) and "sample" in gene_cov:
            items = gene_cov["sample"]
        elif isinstance(gene_cov, list):
            items = gene_cov[:20]
        else:
            # Try cv_stability
            cv = result.get("cv_stability", {})
            if isinstance(cv, dict) and "sample" in cv:
                items = cv["sample"]
            elif isinstance(cv, list):
                items = cv[:20]
            else:
                return None

        if not items:
            return None

        # Items should be dicts with signature_type + some metric
        labels = []
        values = []
        for item in items:
            label = item.get("signature_type", item.get("signature", "?"))
            val = item.get("median_coverage", item.get("median_cv", item.get("value", 0)))
            labels.append(str(label))
            values.append(float(val) if val is not None else 0)

        title = f"Validation Metrics for {atlas} Atlas"

        return {
            "visualization": {
                "type": "bar_chart",
                "title": title,
                "data": {"labels": labels, "values": values},
                "config": {"x_label": "Metric", "y_label": "Value"},
                "container_id": f"viz-{uuid.uuid4().hex[:8]}",
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

        registry_path = Path("/data/parks34/projects/2cytoatlas/docs/registry.json")
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
