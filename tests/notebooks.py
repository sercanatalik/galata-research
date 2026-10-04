import subprocess
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"


def _check(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "marimo", "check", str(path)], capture_output=True, text=True)


def every_notebook_passes_marimo_check():
    # A name defined in two cells stops a notebook from running at all; it
    # reached notebooks/volatility/garch.py twice during the volatility study.
    failing = []
    for path in sorted(NOTEBOOKS.rglob("*.py")):
        run = _check(path)
        if run.returncode != 0 or "critical" in run.stdout:
            first = next((line for line in run.stdout.splitlines() if line.strip()), run.stderr.strip()[:200])
            failing.append(f"{path.relative_to(NOTEBOOKS)}: {first}")
    assert not failing, "marimo check: " + "; ".join(failing)


def the_check_catches_a_name_defined_twice(tmp_path):
    planted = tmp_path / "twice.py"
    planted.write_text(
        "import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    x = 1\n    return (x,)\n\n\n"
        "@app.cell\ndef _():\n    x = 2\n    return (x,)\n\n\nif __name__ == \"__main__\":\n    app.run()\n"
    )
    run = _check(planted)
    assert run.returncode != 0 or "critical" in run.stdout


LIMIT = 1500


def _too_long(paths) -> list[str]:
    return [f"{p.name}: {n} lines" for p in paths if (n := len(p.read_text().splitlines())) > LIMIT]


def no_notebook_grows_past_fifteen_hundred_lines():
    # One notebook per question: liquidity.py reached 2,290 lines and 18 sections before it was split.
    long = _too_long(sorted(NOTEBOOKS.rglob("*.py")))
    assert not long, "notebooks past the limit: " + "; ".join(long)


def every_notebook_lives_in_a_category_folder():
    # The top level holds only the category folders (record, backtests,
    # volatility, correlation, portfolio, liquidity), so a new notebook
    # has to say which kind of study it is.
    stray = sorted(p.name for p in NOTEBOOKS.glob("*.py"))
    assert not stray, "notebooks outside a category folder: " + "; ".join(stray)


def the_size_guard_catches_a_notebook_that_grew(tmp_path):
    grown = tmp_path / "grown.py"
    grown.write_text("x = 1\n" * (LIMIT + 1))
    assert _too_long([grown]) == [f"grown.py: {LIMIT + 1} lines"]
