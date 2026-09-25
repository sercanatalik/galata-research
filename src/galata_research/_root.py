"""Where the record is: `GALATA_VAR`, else `galata-research.toml`, else the sibling checkout.

Resolved on every call, never cached, so a notebook that changes the variable
reads the new root on its next call.
"""

import os
import tomllib
from pathlib import Path

from ._errors import Refused

CONFIG = "galata-research.toml"

# src/galata_research/_root.py → the repository, whose sibling is galata-datawatch.
_REPO = Path(__file__).resolve().parents[2]
_SIBLING = _REPO.parent / "galata-datawatch" / "var"


def root() -> Path:
    """The record's root, which must hold `tape/`."""
    path, source = _resolve()
    if not path.is_dir():
        raise Refused(f"the record's root {path} does not exist (from {source})")
    if not (path / "tape").is_dir():
        raise Refused(f"the record's root {path} has no tape/ (from {source})")
    return path


def _resolve() -> tuple[Path, str]:
    if env := os.environ.get("GALATA_VAR"):
        return Path(env).expanduser(), "GALATA_VAR"
    config = Path.cwd() / CONFIG
    if config.is_file():
        with config.open("rb") as f:
            declared = tomllib.load(f).get("var_root")
        if declared:
            # A relative var_root is relative to the file that declares it.
            return (config.parent / Path(declared).expanduser()).resolve(), f"var_root in {config}"
    return _SIBLING, "the default, ../galata-datawatch/var"
