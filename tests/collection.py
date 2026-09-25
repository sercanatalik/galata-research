import ast
import tomllib
from pathlib import Path

TESTS = Path(__file__).parent


def no_test_function_escapes_collection():
    # pytest here collects only names shaped like claims; a public function of
    # any other shape is silently never run. Helpers start with an underscore.
    with (TESTS.parent / "pyproject.toml").open("rb") as f:
        prefixes = tuple(p.rstrip("*") for p in tomllib.load(f)["tool"]["pytest"]["ini_options"]["python_functions"])
    escaped = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name == "conftest.py":
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith(("_", *prefixes)):
                escaped.append(f"{path.name}:{node.lineno} {node.name}")
    assert not escaped, "never collected: " + ", ".join(escaped)
