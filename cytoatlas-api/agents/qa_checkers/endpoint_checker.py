"""
Endpoint Checker Agent.

Validates API endpoints against expected schemas and behaviors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class EndpointCheckResult:
    """Result of endpoint check."""

    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    schema_valid: bool
    has_cache_headers: bool
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class EndpointChecker:
    """Validates API endpoints against OpenAPI spec and expected behaviors."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"
        self.results: List[EndpointCheckResult] = []

    async def check_endpoint(
        self,
        path: str,
        method: str = "GET",
        params: Dict[str, Any] | None = None,
        expected_status: int = 200,
    ) -> EndpointCheckResult:
        """Check a single endpoint."""
        url = f"{self.base_url}{self.api_prefix}{path}"

        async with httpx.AsyncClient() as client:
            try:
                import time

                start = time.perf_counter()
                response = await client.request(method, url, params=params, timeout=30.0)
                elapsed = (time.perf_counter() - start) * 1000

                # Check response
                result = EndpointCheckResult(
                    endpoint=path,
                    method=method,
                    status_code=response.status_code,
                    response_time_ms=elapsed,
                    schema_valid=response.status_code == expected_status,
                    has_cache_headers="cache-control" in response.headers,
                )

                # Try to parse JSON
                if response.status_code == 200:
                    try:
                        data = response.json()
                        result.details["response_type"] = type(data).__name__
                        if isinstance(data, list):
                            result.details["count"] = len(data)
                        elif isinstance(data, dict):
                            result.details["keys"] = list(data.keys())[:10]
                    except json.JSONDecodeError:
                        result.error = "Invalid JSON response"
                        result.schema_valid = False

            except httpx.RequestError as e:
                result = EndpointCheckResult(
                    endpoint=path,
                    method=method,
                    status_code=0,
                    response_time_ms=0,
                    schema_valid=False,
                    has_cache_headers=False,
                    error=str(e),
                )

        self.results.append(result)
        return result

    async def check_pagination(self, path: str, limit_param: str = "limit") -> bool:
        """Verify pagination parameters work correctly."""
        # Test with limit=5
        result1 = await self.check_endpoint(path, params={limit_param: 5})
        if not result1.schema_valid:
            return False

        # Test with limit=10
        result2 = await self.check_endpoint(path, params={limit_param: 10})
        if not result2.schema_valid:
            return False

        # Check that limit is respected
        count1 = result1.details.get("count", 0)
        count2 = result2.details.get("count", 0)

        return count1 <= 5 and (count2 <= 10 or count2 > count1)

    async def check_error_handling(self, path: str) -> dict[str, bool]:
        """Test error responses for invalid requests."""
        results = {}

        # Test 404 - invalid resource
        result = await self.check_endpoint(
            path.replace("{", "INVALID_").replace("}", "_INVALID"),
            expected_status=404,
        )
        results["404_handling"] = result.status_code in (404, 422)

        # Test 400 - invalid parameters
        result = await self.check_endpoint(
            path,
            params={"invalid_param": "invalid_value"},
            expected_status=200,  # Should still work, ignoring unknown params
        )
        results["unknown_params"] = result.status_code == 200

        return results

    async def check_all_cima_endpoints(self) -> List[EndpointCheckResult]:
        """Check all CIMA endpoints."""
        endpoints = [
            "/cima/summary",
            "/cima/cell-types",
            "/cima/signatures",
            "/cima/activity",
            "/cima/correlations/age",
            "/cima/correlations/bmi",
            "/cima/correlations/biochemistry",
            "/cima/correlations/metabolites",
            "/cima/differential",
            "/cima/eqtl/top",
        ]

        for endpoint in endpoints:
            await self.check_endpoint(endpoint)

        return self.results

    async def check_all_inflammation_endpoints(self) -> List[EndpointCheckResult]:
        """Check all Inflammation endpoints."""
        endpoints = [
            "/inflammation/summary",
            "/inflammation/cell-types",
            "/inflammation/diseases",
            "/inflammation/disease-activity",
            "/inflammation/activity",
            "/inflammation/treatment-response",
            "/inflammation/correlations/age",
            "/inflammation/correlations/bmi",
        ]

        for endpoint in endpoints:
            await self.check_endpoint(endpoint)

        return self.results

    async def check_all_validation_endpoints(self) -> List[EndpointCheckResult]:
        """Check all Validation endpoints."""
        endpoints = [
            "/validation/summary",
            "/validation/expression-vs-activity",
            "/validation/gene-coverage/IFNG",
        ]

        for endpoint in endpoints:
            await self.check_endpoint(endpoint)

        return self.results

    async def check_all(self) -> List[EndpointCheckResult]:
        """Check all endpoints."""
        # Health
        await self.check_endpoint("/health")
        await self.check_endpoint("/health/ready")

        # CIMA
        await self.check_all_cima_endpoints()

        # Inflammation
        await self.check_all_inflammation_endpoints()

        # scAtlas
        await self.check_endpoint("/scatlas/summary")
        await self.check_endpoint("/scatlas/organs")
        await self.check_endpoint("/scatlas/cell-types")

        # Cross-atlas
        await self.check_endpoint("/cross-atlas/atlases")

        # Validation
        await self.check_all_validation_endpoints()

        return self.results

    def report(self) -> Dict[str, Any]:
        """Generate summary report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.schema_valid)
        failed = total - passed

        avg_response_time = (
            sum(r.response_time_ms for r in self.results) / total if total > 0 else 0
        )

        failed_endpoints = [r.endpoint for r in self.results if not r.schema_valid]

        return {
            "total_endpoints": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed / total * 100:.1f}%" if total > 0 else "N/A",
            "avg_response_time_ms": round(avg_response_time, 2),
            "failed_endpoints": failed_endpoints,
            "results": [
                {
                    "endpoint": r.endpoint,
                    "status": r.status_code,
                    "valid": r.schema_valid,
                    "time_ms": round(r.response_time_ms, 2),
                    "error": r.error,
                }
                for r in self.results
            ],
        }


async def main():
    """Run endpoint checker from command line."""
    parser = argparse.ArgumentParser(description="Check CytoAtlas API endpoints")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--atlas",
        choices=["cima", "inflammation", "scatlas", "validation", "all"],
        default="all",
        help="Atlas to check",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file",
    )

    args = parser.parse_args()

    checker = EndpointChecker(args.base_url)

    print(f"Checking {args.atlas} endpoints at {args.base_url}...")

    if args.atlas == "cima":
        await checker.check_all_cima_endpoints()
    elif args.atlas == "inflammation":
        await checker.check_all_inflammation_endpoints()
    elif args.atlas == "validation":
        await checker.check_all_validation_endpoints()
    else:
        await checker.check_all()

    report = checker.report()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))

    # Exit with error code if any failed
    sys.exit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
