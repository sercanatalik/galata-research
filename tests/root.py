import pytest

from galata_research import Refused, _root


def _record(path):
    (path / "tape").mkdir(parents=True)
    return path


def the_environment_variable_wins(tmp_path, monkeypatch):
    from_env = _record(tmp_path / "env")
    from_file = _record(tmp_path / "file")
    (tmp_path / "galata-research.toml").write_text(f'var_root = "{from_file}"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GALATA_VAR", str(from_env))
    assert _root.root() == from_env


def a_relative_var_root_is_relative_to_its_file(tmp_path, monkeypatch):
    _record(tmp_path / "data")
    (tmp_path / "galata-research.toml").write_text('var_root = "data"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GALATA_VAR", raising=False)
    assert _root.root() == (tmp_path / "data").resolve()


def a_missing_root_is_refused_by_name(tmp_path, monkeypatch):
    missing = tmp_path / "nowhere"
    monkeypatch.setenv("GALATA_VAR", str(missing))
    with pytest.raises(Refused, match=r"nowhere.*does not exist.*GALATA_VAR"):
        _root.root()


def a_root_without_a_tape_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("GALATA_VAR", str(tmp_path))
    with pytest.raises(Refused, match="has no tape/"):
        _root.root()


def a_changed_variable_is_seen_by_the_next_call(tmp_path, monkeypatch):
    first, second = _record(tmp_path / "a"), _record(tmp_path / "b")
    monkeypatch.setenv("GALATA_VAR", str(first))
    assert _root.root() == first
    monkeypatch.setenv("GALATA_VAR", str(second))
    assert _root.root() == second
