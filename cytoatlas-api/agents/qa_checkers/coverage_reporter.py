"""
Coverage Reporter Agent.

Tracks API endpoint implementation progress and generates coverage reports.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class EndpointInfo:
    """Information about an API endpoint."""

    path: str
    method: str
    function_name: str
    file: str
    has_docstring: bool
    has_tests: bool = False


@dataclass
class CoverageReport:
    """Coverage report data."""

    total_endpoints: int
    documented: int
    tested: int
    by_router: Dict[str, List[EndpointInfo]] = field(default_factory=dict)


class CoverageReporter:
    """Reports on API endpoint coverage and implementation status."""

    EXPECTED_ENDPOINTS = {
        "health": [
            ("GET", "/health"),
            ("GET", "/health/ready"),
        ],
        "cima": [
            ("GET", "/cima/summary"),
            ("GET", "/cima/cell-types"),
            ("GET", "/cima/signatures"),
            ("GET", "/cima/activity"),
            ("GET", "/cima/correlations/age"),
            ("GET", "/cima/correlations/bmi"),
            ("GET", "/cima/correlations/biochemistry"),
            ("GET", "/cima/correlations/metabolites"),
            ("GET", "/cima/differential"),
            ("GET", "/cima/eqtl"),
            ("GET", "/cima/eqtl/top"),
            ("GET", "/cima/boxplots/age/{signature}"),
            ("GET", "/cima/boxplots/bmi/{signature}"),
        ],
        "inflammation": [
            ("GET", "/inflammation/summary"),
            ("GET", "/inflammation/cell-types"),
            ("GET", "/inflammation/diseases"),
            ("GET", "/inflammation/disease-activity"),
            ("GET", "/inflammation/activity"),
            ("GET", "/inflammation/treatment-response"),
            ("GET", "/inflammation/roc-curves"),
            ("GET", "/inflammation/feature-importance"),
            ("GET", "/inflammation/cohort-validation"),
            ("GET", "/inflammation/disease-sankey"),
            ("GET", "/inflammation/correlations/age"),
            ("GET", "/inflammation/correlations/bmi"),
        ],
        "scatlas": [
            ("GET", "/scatlas/summary"),
            ("GET", "/scatlas/organs"),
            ("GET", "/scatlas/cell-types"),
            ("GET", "/scatlas/cancer-comparison"),
            ("GET", "/scatlas/cancer-types"),
        ],
        "cross_atlas": [
            ("GET", "/cross-atlas/atlases"),
            ("GET", "/cross-atlas/comparison"),
            ("GET", "/cross-atlas/conserved-signatures"),
        ],
        "atlases": [
            ("GET", "/atlases"),
            ("POST", "/atlases/register"),
            ("GET", "/atlases/{atlas_name}"),
            ("DELETE", "/atlases/{atlas_name}"),
            ("GET", "/atlases/{atlas_name}/summary"),
            ("GET", "/atlases/{atlas_name}/cell-types"),
            ("GET", "/atlases/{atlas_name}/signatures"),
            ("GET", "/atlases/{atlas_name}/features"),
            ("GET", "/atlases/{atlas_name}/activity"),
            ("GET", "/atlases/{atlas_name}/correlations/{variable}"),
            ("GET", "/atlases/{atlas_name}/differential"),
            ("GET", "/atlases/{atlas_name}/diseases"),
            ("GET", "/atlases/{atlas_name}/organs"),
        ],
        "validation": [
            ("GET", "/validation/summary"),
            ("GET", "/validation/sample-level/{atlas}/{signature}"),
            ("GET", "/validation/celltype-level/{atlas}/{signature}"),
            ("GET", "/validation/pseudobulk-vs-singlecell/{atlas}/{signature}"),
            ("GET", "/validation/singlecell-direct/{atlas}/{signature}"),
            ("GET", "/validation/biological-associations/{atlas}"),
            ("GET", "/validation/gene-coverage/{atlas}/{signature}"),
            ("GET", "/validation/cv-stability/{atlas}"),
        ],
        "export": [
            ("GET", "/export/csv/{data_type}"),
            ("GET", "/export/json/{data_type}"),
        ],
    }

    def __init__(self, routers_path: Optional[Path] = None, tests_path: Optional[Path] = None):
        self.routers_path = routers_path or Path(__file__).parent.parent.parent / "app" / "routers"
        self.tests_path = tests_path or Path(__file__).parent.parent.parent / "tests"
        self.endpoints: Dict[str, List[EndpointInfo]] = {}

    def _parse_router_file(self, file_path: Path) -> List[EndpointInfo]:
        """Parse a router file to extract endpoint information."""
        endpoints: List[EndpointInfo] = []

        try:
            with open(file_path) as f:
                source = f.read()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for route decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            # Get decorator name (e.g., router.get, router.post)
                            if isinstance(decorator.func, ast.Attribute):
                                method = decorator.func.attr.upper()
                                if method in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                                    # Get path from first argument
                                    path = ""
                                    if decorator.args:
                                        if isinstance(decorator.args[0], ast.Constant):
                                            path = decorator.args[0].value

                                    # Check for docstring
                                    has_docstring = (
                                        isinstance(node.body[0], ast.Expr)
                                        and isinstance(node.body[0].value, ast.Constant)
                                        if node.body
                                        else False
                                    )

                                    endpoints.append(
                                        EndpointInfo(
                                            path=path,
                                            method=method,
                                            function_name=node.name,
                                            file=file_path.name,
                                            has_docstring=has_docstring,
                                        )
                                    )
        except (SyntaxError, FileNotFoundError):
            pass

        return endpoints

    def scan_routers(self) -> Dict[str, List[EndpointInfo]]:
        """Scan all router files and extract endpoints."""
        self.endpoints = {}

        if not self.routers_path.exists():
            return self.endpoints

        for router_file in self.routers_path.glob("*.py"):
            if router_file.name.startswith("_"):
                continue

            router_name = router_file.stem
            endpoints = self._parse_router_file(router_file)

            if endpoints:
                self.endpoints[router_name] = endpoints

        return self.endpoints

    def _check_tests_exist(self, router_name: str) -> Set[str]:
        """Check which endpoints have tests."""
        tested_functions: Set[str] = set()

        test_file = self.tests_path / "unit" / f"test_{router_name}.py"
        if not test_file.exists():
            test_file = self.tests_path / "integration" / f"test_{router_name}.py"

        if test_file.exists():
            try:
                with open(test_file) as f:
                    source = f.read()

                # Simple pattern matching for test functions
                import re

                test_functions = re.findall(r"def test_(\w+)", source)
                tested_functions = set(test_functions)
            except (SyntaxError, FileNotFoundError):
                pass

        return tested_functions

    def report(self) -> CoverageReport:
        """Generate coverage report."""
        if not self.endpoints:
            self.scan_routers()

        total = 0
        documented = 0
        tested = 0

        for router_name, endpoints in self.endpoints.items():
            tested_functions = self._check_tests_exist(router_name)

            for endpoint in endpoints:
                total += 1
                if endpoint.has_docstring:
                    documented += 1
                if endpoint.function_name in tested_functions:
                    endpoint.has_tests = True
                    tested += 1

        return CoverageReport(
            total_endpoints=total,
            documented=documented,
            tested=tested,
            by_router=self.endpoints,
        )

    def list_missing(self) -> Dict[str, List[Tuple[str, str]]]:
        """List endpoints that are expected but not implemented."""
        if not self.endpoints:
            self.scan_routers()

        missing: Dict[str, List[Tuple[str, str]]] = {}

        for router_name, expected in self.EXPECTED_ENDPOINTS.items():
            implemented_paths = set()

            if router_name in self.endpoints:
                for endpoint in self.endpoints[router_name]:
                    implemented_paths.add((endpoint.method, endpoint.path))

            # Find missing
            router_missing = []
            for method, path in expected:
                # Normalize path comparison (ignore prefix variations)
                found = False
                for impl_method, impl_path in implemented_paths:
                    if impl_method == method and (impl_path == path or impl_path.endswith(path)):
                        found = True
                        break

                if not found:
                    router_missing.append((method, path))

            if router_missing:
                missing[router_name] = router_missing

        return missing

    def to_json(self) -> str:
        """Export report as JSON."""
        report = self.report()
        missing = self.list_missing()

        return json.dumps(
            {
                "summary": {
                    "total_endpoints": report.total_endpoints,
                    "documented": report.documented,
                    "tested": report.tested,
                    "documentation_rate": (
                        f"{report.documented / report.total_endpoints * 100:.1f}%"
                        if report.total_endpoints > 0
                        else "N/A"
                    ),
                    "test_coverage": (
                        f"{report.tested / report.total_endpoints * 100:.1f}%"
                        if report.total_endpoints > 0
                        else "N/A"
                    ),
                },
                "by_router": {
                    name: [
                        {
                            "method": e.method,
                            "path": e.path,
                            "function": e.function_name,
                            "documented": e.has_docstring,
                            "tested": e.has_tests,
                        }
                        for e in endpoints
                    ]
                    for name, endpoints in report.by_router.items()
                },
                "missing_endpoints": missing,
            },
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report CytoAtlas API coverage")
    parser.add_argument(
        "--routers-path",
        type=Path,
        help="Path to routers directory",
    )
    parser.add_argument(
        "--tests-path",
        type=Path,
        help="Path to tests directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only show missing endpoints",
    )

    args = parser.parse_args()

    reporter = CoverageReporter(
        routers_path=args.routers_path,
        tests_path=args.tests_path,
    )

    if args.missing_only:
        missing = reporter.list_missing()
        output = json.dumps(missing, indent=2)
    else:
        output = reporter.to_json()

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)
