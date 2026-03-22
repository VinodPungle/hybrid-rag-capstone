"""
Structured Logging & LLM Call Tracking (Step 2: Logging & Exception Handling)

Replaces all print() statements across the project with structured logging.
Provides LLMOps-specific features:
  - Token usage tracking (prompt/completion tokens per LLM call)
  - Latency measurement for each LLM API call
  - Optional prompt logging for debugging

All behavior is controlled by flags in config.yaml → llmops.monitoring:
  - track_token_usage: log token counts per LLM call
  - track_latency: log response time per LLM call
  - log_prompts: log full prompt text (disabled by default for privacy)

Log output format:
    2026-03-22 10:41:18 | INFO    | llm.generator | LLM call | {"call_type": "generation", ...}
"""

import logging
import time
import json
import sys
from functools import wraps
from config.settings import get as cfg

# Read monitoring flags from config.yaml → llmops.monitoring
_monitoring = cfg("llmops", "monitoring") or {}
TRACK_TOKEN_USAGE = _monitoring.get("track_token_usage", False)
TRACK_LATENCY = _monitoring.get("track_latency", False)
LOG_PROMPTS = _monitoring.get("log_prompts", False)


def get_logger(name):
    """
    Return a structured logger for the given module name.

    Creates a logger with a consistent format across all modules:
    timestamp | level | module | message

    Args:
        name: Module name, typically __name__ (e.g., "llm.generator")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_llm_call(logger, response, call_type="generation", latency_s=None):
    """
    Log token usage and latency from an Azure OpenAI response.

    Only logs fields that are enabled in config.yaml → llmops.monitoring.
    Called after every LLM API call (generation and entity extraction).

    Args:
        logger: The module's logger instance
        response: OpenAI-style response object with .usage attribute
        call_type: "generation" or "entity_extraction"
        latency_s: Wall-clock time for the API call in seconds
    """
    extra = {"call_type": call_type}

    # Include latency if tracking is enabled
    if TRACK_LATENCY and latency_s is not None:
        extra["latency_s"] = round(latency_s, 3)

    # Include token counts if tracking is enabled and response has usage data
    if TRACK_TOKEN_USAGE and hasattr(response, "usage") and response.usage:
        extra["prompt_tokens"] = response.usage.prompt_tokens
        extra["completion_tokens"] = response.usage.completion_tokens
        extra["total_tokens"] = response.usage.total_tokens

    logger.info("LLM call | %s", json.dumps(extra))


def timed(fn):
    """Decorator that measures wall-clock time and returns (result, elapsed_s)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper
