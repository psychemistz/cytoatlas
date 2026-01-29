"""
CytoAtlas QA Agents.

Multi-agent pipeline for validating API endpoints, tracking progress,
and generating validation data.

Usage:
    from agents.progress_tracker import ProgressTracker
    from agents.validation_generator import ValidationDataGenerator
    from agents.qa_checkers.endpoint_checker import EndpointChecker
    from agents.qa_checkers.coverage_reporter import CoverageReporter
    from agents.qa_checkers.schema_validator import SchemaValidator
"""

__all__ = [
    "ProgressTracker",
    "ValidationDataGenerator",
]


def __getattr__(name: str):
    """Lazy import to avoid dependency issues."""
    if name == "ProgressTracker":
        from agents.progress_tracker import ProgressTracker
        return ProgressTracker
    elif name == "ValidationDataGenerator":
        from agents.validation_generator import ValidationDataGenerator
        return ValidationDataGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
