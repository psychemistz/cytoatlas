"""
QA Checker Agents.

Agents for validating API endpoints and tracking coverage.

Usage:
    from agents.qa_checkers.endpoint_checker import EndpointChecker
    from agents.qa_checkers.schema_validator import SchemaValidator
    from agents.qa_checkers.coverage_reporter import CoverageReporter
"""

__all__ = [
    "EndpointChecker",
    "SchemaValidator",
    "CoverageReporter",
]


def __getattr__(name: str):
    """Lazy import to avoid dependency issues."""
    if name == "EndpointChecker":
        from agents.qa_checkers.endpoint_checker import EndpointChecker
        return EndpointChecker
    elif name == "SchemaValidator":
        from agents.qa_checkers.schema_validator import SchemaValidator
        return SchemaValidator
    elif name == "CoverageReporter":
        from agents.qa_checkers.coverage_reporter import CoverageReporter
        return CoverageReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
