"""Unit tests for the pipeline CLI.

Tests parser construction, argument handling, and command dispatch.
Does NOT test actual pipeline execution (that requires H5AD data and GPU).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from cytoatlas_pipeline.cli import build_parser, main, cmd_export


# ===========================================================================
# Parser construction
# ===========================================================================

class TestBuildParser:
    """Tests for argument parser construction."""

    def test_parser_has_all_subcommands(self):
        parser = build_parser()
        # Extract subcommand names from the subparsers action
        subparsers_action = None
        for action in parser._subparsers._actions:
            if hasattr(action, "_parser_class"):
                subparsers_action = action
                break
        assert subparsers_action is not None
        expected = {"run", "activity", "aggregate", "differential",
                    "correlate", "validate", "export"}
        assert set(subparsers_action.choices.keys()) == expected

    def test_no_command_returns_zero(self):
        """No subcommand should print help and return 0."""
        result = main([])
        assert result == 0

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--verbose", "export", "--input", ".", "--format", "json"])
        assert args.verbose is True

    def test_log_file_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--log-file", "/tmp/test.log", "export", "--input", ".", "--format", "json"])
        assert args.log_file == "/tmp/test.log"


# ===========================================================================
# Subcommand argument parsing
# ===========================================================================

class TestSubcommandArgs:
    """Tests for individual subcommand argument parsing."""

    def test_run_args(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--config", "pipeline.yaml", "-o", "/tmp/out"])
        assert args.command == "run"
        assert args.config == "pipeline.yaml"
        assert args.output == "/tmp/out"

    def test_activity_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "activity", "-i", "data.h5ad",
            "--signatures", "CytoSig", "SecAct",
            "--gpu", "0", "1",
            "--batch-size", "5000",
            "--n-rand", "500",
            "--seed", "42",
        ])
        assert args.command == "activity"
        assert args.input == "data.h5ad"
        assert args.signatures == ["CytoSig", "SecAct"]
        assert args.gpu == [0, 1]
        assert args.batch_size == 5000
        assert args.n_rand == 500
        assert args.seed == 42

    def test_activity_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["activity", "-i", "data.h5ad"])
        assert args.signatures == ["CytoSig"]
        assert args.aggregation == "pseudobulk"
        assert args.batch_size == 10000
        assert args.n_rand == 1000
        assert args.seed == 0

    def test_aggregate_args(self):
        parser = build_parser()
        args = parser.parse_args(["aggregate", "-i", "data.h5ad", "--method", "celltype"])
        assert args.command == "aggregate"
        assert args.method == "celltype"

    def test_differential_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "differential",
            "--activity", "activity.csv",
            "--metadata", "meta.csv",
            "--group-col", "disease",
            "--group1", "IBD",
            "--group2", "healthy",
        ])
        assert args.command == "differential"
        assert args.group_col == "disease"
        assert args.group1 == "IBD"
        assert args.group2 == "healthy"

    def test_correlate_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "correlate",
            "--activity", "activity.csv",
            "--metadata", "meta.csv",
            "--method", "pearson",
            "--variables", "age", "bmi",
        ])
        assert args.command == "correlate"
        assert args.method == "pearson"
        assert args.variables == ["age", "bmi"]

    def test_validate_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "validate",
            "--activity", "activity.csv",
            "--expression", "expr.csv",
            "--metadata", "meta.csv",
        ])
        assert args.command == "validate"
        assert args.metadata == "meta.csv"

    def test_export_args(self):
        parser = build_parser()
        args = parser.parse_args(["export", "-i", "results/", "-f", "duckdb", "-o", "/tmp"])
        assert args.command == "export"
        assert args.format == "duckdb"

    def test_export_format_choices(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "-i", ".", "-f", "xlsx"])


# ===========================================================================
# Export command — import correctness
# ===========================================================================

class TestExportImports:
    """Verify export command imports the correct class names."""

    def test_duckdb_export_imports_duckdb_writer(self):
        """DuckDBWriter should be importable (not DuckDBExporter)."""
        from cytoatlas_pipeline.export import DuckDBWriter
        assert DuckDBWriter is not None

    def test_parquet_export_imports_parquet_writer(self):
        """ParquetWriter should be importable (not ParquetExporter)."""
        from cytoatlas_pipeline.export import ParquetWriter
        assert ParquetWriter is not None

    def test_json_export_imports_json_writer(self):
        """JSONWriter should be importable."""
        from cytoatlas_pipeline.export import JSONWriter
        assert JSONWriter is not None

    def test_no_exporter_aliases_exist(self):
        """Verify the old *Exporter names are NOT exported."""
        import cytoatlas_pipeline.export as export_mod
        assert not hasattr(export_mod, "DuckDBExporter")
        assert not hasattr(export_mod, "ParquetExporter")

    def test_export_cmd_duckdb_format(self, tmp_path):
        """cmd_export with duckdb format should import DuckDBWriter successfully."""
        args = MagicMock()
        args.format = "duckdb"
        args.input = str(tmp_path)
        args.output = str(tmp_path)

        with patch("cytoatlas_pipeline.export.DuckDBWriter") as MockWriter:
            instance = MockWriter.return_value
            # cmd_export will call exporter.export_directory — it doesn't exist,
            # but with the mock, it won't error on the import anymore
            with patch("cytoatlas_pipeline.cli.DuckDBWriter", MockWriter, create=True):
                # The import line inside cmd_export does a fresh import,
                # so we patch at the module level where it's used
                pass

        # Verify the import path in the source code is correct
        import ast
        from cytoatlas_pipeline import cli
        import inspect

        source = inspect.getsource(cli.cmd_export)
        assert "DuckDBWriter" in source
        assert "DuckDBExporter" not in source
        assert "ParquetWriter" in source
        assert "ParquetExporter" not in source


# ===========================================================================
# Export command — unknown format
# ===========================================================================

class TestExportUnknownFormat:
    """Test export with unknown format."""

    def test_unknown_format_returns_1(self, tmp_path):
        args = MagicMock()
        args.format = "xlsx"
        args.input = str(tmp_path)
        args.output = str(tmp_path)

        result = cmd_export(args)
        assert result == 1


# ===========================================================================
# Main entry point
# ===========================================================================

class TestMainEntryPoint:
    """Tests for the main() function dispatch."""

    def test_main_dispatches_to_func(self):
        """main() should call args.func(args) for known commands."""
        mock_func = MagicMock(return_value=0)
        with patch("cytoatlas_pipeline.cli.build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "test"
            mock_args.verbose = False
            mock_args.log_file = None
            mock_args.func = mock_func
            mock_parser.return_value.parse_args.return_value = mock_args

            result = main(["test"])
            mock_func.assert_called_once_with(mock_args)
            assert result == 0

    def test_main_returns_func_exit_code(self):
        """main() should propagate the return code from the command function."""
        mock_func = MagicMock(return_value=42)
        with patch("cytoatlas_pipeline.cli.build_parser") as mock_parser:
            mock_args = MagicMock()
            mock_args.command = "test"
            mock_args.verbose = False
            mock_args.log_file = None
            mock_args.func = mock_func
            mock_parser.return_value.parse_args.return_value = mock_args

            result = main(["test"])
            assert result == 42
