"""Where the reference store is: `GALATA_REFERENCE`, else `galata-research.toml`, else `var/reference/`.

Resolved on every call, never cached, as the record's root is. The store is
apart from the record: research reads the record and writes only here.
"""

import os
import tomllib
from pathlib import Path

from .. import _root
from .._errors import Refused

ENV = "GALATA_REFERENCE"
KEY = "reference_root"
_DEFAULT = _root._REPO / "var" / "reference"


def resolve() -> tuple[Path, str]:
    """The store's path and where it came from, whether or not it exists yet."""
    if env := os.environ.get(ENV):
        return Path(env).expanduser(), ENV
    config = Path.cwd() / _root.CONFIG
    if config.is_file():
        with config.open("rb") as f:
            declared = tomllib.load(f).get(KEY)
        if declared:
            # A relative reference_root is relative to the file that declares it.
            return (config.parent / Path(declared).expanduser()).resolve(), f"{KEY} in {config}"
    return _DEFAULT, "the default, var/reference/"


def root() -> Path:
    """The reference store's root, which must exist."""
    path, source = resolve()
    if not path.is_dir():
        raise Refused(f"the reference store {path} does not exist (from {source}); fetch into it with galata-fetch")
    return path
