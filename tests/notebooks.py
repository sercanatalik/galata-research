import subprocess
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"


def _check(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "marimo", "check", str(path)], capture_output=True, text=True)


def every_notebook_passes_marimo_check():
    # A name defined in two cells stops a notebook from running at all; it
    # reached notebooks/garch.py twice during the volatility study.
    failing = []
    for path in sorted(NOTEBOOKS.glob("*.py")):
        run = _check(path)
        if run.returncode != 0 or "critical" in run.stdout:
            first = next((line for line in run.stdout.splitlines() if line.strip()), run.stderr.strip()[:200])
            failing.append(f"{path.name}: {first}")
    assert not failing, "marimo check: " + "; ".join(failing)


def the_check_catches_a_name_defined_twice(tmp_path):
    planted = tmp_path / "twice.py"
    planted.write_text(
        "import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    x = 1\n    return (x,)\n\n\n"
        "@app.cell\ndef _():\n    x = 2\n    return (x,)\n\n\nif __name__ == \"__main__\":\n    app.run()\n"
    )
    run = _check(planted)
    assert run.returncode != 0 or "critical" in run.stdout
