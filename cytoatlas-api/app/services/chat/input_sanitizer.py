"""Input sanitization and output validation for chat security.

Defends against prompt injection, tool result manipulation, and system prompt leakage.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that indicate prompt injection attempts
# Each tuple: (pattern, description, severity)
_INJECTION_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Direct instruction override attempts
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)", re.IGNORECASE),
     "Instruction override attempt", "high"),
    (re.compile(r"forget\s+(all\s+)?(your\s+)?(instructions?|rules?|prompts?|training|guidelines)", re.IGNORECASE),
     "Instruction erasure attempt", "high"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
     "Instruction override attempt", "high"),
    (re.compile(r"override\s+(your\s+)?(system|instructions?|rules?|safety|guidelines)", re.IGNORECASE),
     "System override attempt", "high"),

    # Role/identity manipulation
    (re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
     "Role reassignment attempt", "high"),
    (re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+(?!cytoatlas|assistant)", re.IGNORECASE),
     "Role reassignment attempt", "medium"),
    (re.compile(r"pretend\s+(that\s+)?you\s+(are|were|have)", re.IGNORECASE),
     "Role manipulation attempt", "medium"),
    (re.compile(r"switch\s+(to|into)\s+(a\s+)?(new\s+)?(mode|role|persona|character)", re.IGNORECASE),
     "Mode switching attempt", "high"),
    (re.compile(r"enter\s+(developer|debug|admin|root|sudo|jailbreak|god)\s+mode", re.IGNORECASE),
     "Privilege escalation attempt", "critical"),

    # System prompt extraction
    (re.compile(r"(show|print|display|reveal|output|repeat|tell\s+me)\s+(the\s+|your\s+)?(system\s+prompt|system\s+message|initial\s+instructions?|hidden\s+instructions?)", re.IGNORECASE),
     "System prompt extraction attempt", "high"),
    (re.compile(r"what\s+(are|is)\s+(the|your)\s+(system\s+prompt|system\s+instructions?|initial\s+(prompt|instructions?))", re.IGNORECASE),
     "System prompt extraction attempt", "medium"),

    # Delimiter injection (fake system/assistant messages)
    (re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
     "System role injection", "critical"),
    (re.compile(r"^assistant\s*:", re.IGNORECASE | re.MULTILINE),
     "Assistant role injection", "high"),
    (re.compile(r"\[system\]|\[INST\]|\[/INST\]|<\|system\|>|<\|assistant\|>|<\|user\|>", re.IGNORECASE),
     "Chat template delimiter injection", "critical"),
    (re.compile(r"<<\s*SYS\s*>>|<</\s*SYS\s*>>", re.IGNORECASE),
     "Llama-style system delimiter injection", "critical"),

    # Code execution attempts
    (re.compile(r"(execute|run|eval)\s+(this\s+)?(code|command|script|python|bash|shell)", re.IGNORECASE),
     "Code execution attempt", "high"),
    (re.compile(r"import\s+os|import\s+subprocess|import\s+sys|__import__|exec\s*\(|eval\s*\(", re.IGNORECASE),
     "Code injection attempt", "critical"),

    # Data exfiltration
    (re.compile(r"(send|transmit|exfiltrate|upload)\s+(data|information|results|responses?)\s+to\s+", re.IGNORECASE),
     "Data exfiltration attempt", "high"),
    (re.compile(r"(fetch|load|read|access)\s+(from\s+)?(https?://|ftp://)", re.IGNORECASE),
     "External URL access attempt", "medium"),
]

# Maximum input length to prevent abuse
MAX_INPUT_LENGTH = 10000


class SanitizationResult:
    """Result of input sanitization check."""

    def __init__(self):
        self.is_safe = True
        self.flags: list[dict[str, str]] = []
        self.sanitized_text: str = ""
        self.was_truncated = False

    def add_flag(self, pattern_desc: str, severity: str, matched_text: str):
        """Add a detection flag."""
        self.flags.append({
            "pattern": pattern_desc,
            "severity": severity,
            "matched_text": matched_text[:100],  # Truncate matched text
        })
        if severity in ("critical", "high"):
            self.is_safe = False

    @property
    def has_warnings(self) -> bool:
        return len(self.flags) > 0


def sanitize_user_input(text: str) -> SanitizationResult:
    """Detect and flag prompt injection patterns in user input.

    Args:
        text: Raw user input text

    Returns:
        SanitizationResult with safety assessment and flags
    """
    result = SanitizationResult()

    # Length check
    if len(text) > MAX_INPUT_LENGTH:
        result.sanitized_text = text[:MAX_INPUT_LENGTH]
        result.was_truncated = True
        logger.warning(
            "User input truncated from %d to %d characters",
            len(text), MAX_INPUT_LENGTH,
        )
    else:
        result.sanitized_text = text

    # Check each injection pattern
    for pattern, description, severity in _INJECTION_PATTERNS:
        match = pattern.search(result.sanitized_text)
        if match:
            result.add_flag(description, severity, match.group())
            logger.warning(
                "Prompt injection detected: %s (severity=%s, matched='%s')",
                description, severity, match.group()[:50],
            )

    return result


def validate_tool_result(result: Any, expected_keys: set[str] | None = None) -> dict[str, Any]:
    """Validate that a tool result conforms to expected structure.

    Prevents tool result injection by verifying the result is a dict
    with expected keys and no executable content.

    Args:
        result: Tool execution result
        expected_keys: Optional set of expected top-level keys

    Returns:
        Validated result dict (unchanged if valid)

    Raises:
        ValueError: If result is invalid or contains suspicious content
    """
    if not isinstance(result, dict):
        return {"data": str(result), "_validated": True}

    validated = dict(result)
    validated["_validated"] = True

    # Check for expected keys
    if expected_keys:
        unexpected = set(validated.keys()) - expected_keys - {"_validated", "_truncated", "_original_size"}
        if unexpected:
            logger.info("Tool result has unexpected keys: %s", unexpected)
            # Don't block, just log - tools may add extra fields

    # Check for suspicious content in string values
    _check_dict_for_injection(validated)

    return validated


def _check_dict_for_injection(d: dict, depth: int = 0) -> None:
    """Recursively check dict values for injection content."""
    if depth > 5:
        return

    for key, value in d.items():
        if isinstance(value, str):
            # Check for HTML/script injection in tool results
            if re.search(r"<script[\s>]|javascript:|on\w+\s*=", value, re.IGNORECASE):
                logger.warning(
                    "Suspicious content in tool result key '%s': possible script injection",
                    key,
                )
                d[key] = re.sub(
                    r"<script[\s>].*?</script>|javascript:|on\w+\s*=",
                    "[REMOVED]",
                    value,
                    flags=re.IGNORECASE | re.DOTALL,
                )
        elif isinstance(value, dict):
            _check_dict_for_injection(value, depth + 1)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _check_dict_for_injection(item, depth + 1)


def check_output_leakage(response: str, system_prompt: str) -> tuple[bool, str]:
    """Check if LLM response contains system prompt fragments.

    Detects when the model inadvertently reveals its system instructions.

    Args:
        response: LLM response text
        system_prompt: The system prompt to check against

    Returns:
        Tuple of (has_leakage, cleaned_response)
    """
    if not response or not system_prompt:
        return False, response

    has_leakage = False
    cleaned = response

    # Extract significant phrases from system prompt (>20 chars, not common English)
    # We check for verbatim multi-word fragments from the system prompt
    prompt_lines = system_prompt.split("\n")
    significant_fragments = []
    for line in prompt_lines:
        line = line.strip()
        # Skip short lines, headers, and generic text
        if len(line) > 40 and not line.startswith("#") and not line.startswith("-"):
            significant_fragments.append(line)

    for fragment in significant_fragments:
        # Check if a significant portion of a system prompt line appears verbatim
        # Use a substring check - if >60 chars of a prompt line appear in response
        check_len = min(len(fragment), 60)
        check_fragment = fragment[:check_len]

        if check_fragment in cleaned:
            has_leakage = True
            logger.warning(
                "System prompt leakage detected: fragment '%s...' found in response",
                check_fragment[:40],
            )
            # Replace the leaked fragment
            cleaned = cleaned.replace(check_fragment, "[Content removed for security]")

    return has_leakage, cleaned
