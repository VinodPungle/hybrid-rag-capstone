"""
Centralized Configuration Loader (Step 1: YAML Configuration)

Reads all tunable, non-secret parameters from config/config.yaml and
provides them to every module in the project via the get() function.
This replaces hardcoded values across the codebase with a single
source of truth for reproducibility (a core LLMOps practice).

Secrets (API keys, passwords) remain in .env and are NOT stored here.

Usage:
    from config.settings import get as cfg

    # Get an entire section as a dict
    ingestion_cfg = cfg("ingestion")

    # Get a specific key from a section
    chunk_size = cfg("ingestion", "chunk_size")
"""

import os
import yaml

# Path to the YAML config file, resolved relative to this file's directory
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Module-level cache: loaded once, reused on every get() call
_config = None


def _load():
    """Lazy-load and cache the YAML config file (singleton pattern)."""
    global _config
    if _config is None:
        with open(_CONFIG_PATH, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get(section, key=None):
    """
    Return a config section dict, or a specific key within it.

    Args:
        section: Top-level section name (e.g., "ingestion", "llm", "retrieval")
        key: Optional specific key within the section

    Returns:
        dict if key is None, or the value of the specific key.
        Returns {} for missing sections, None for missing keys.
    """
    cfg = _load()
    section_data = cfg.get(section, {})
    if key is None:
        return section_data
    return section_data.get(key)
