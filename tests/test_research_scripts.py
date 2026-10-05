"""Every research script must at least import: catches names missing from the package (e.g. a file
that git silently failed to track). Scripts needing the optional RL libraries are skipped without them."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = sorted(Path(__file__).resolve().parents[1].joinpath("research").glob("*.py"))
OPTIONAL = {"torch", "gymnasium", "stable_baselines3", "matplotlib"}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_research_script_imports(path):
    spec = importlib.util.spec_from_file_location(f"research_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # top-level imports and definitions only; main() is not run
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in OPTIONAL:
            pytest.skip(f"optional dependency missing: {exc.name}")
        raise
    assert hasattr(module, "main")


def test_package_code_is_not_gitignored():
    """The data package once matched a 'data/' ignore rule and silently disappeared from git."""
    ignore = Path(__file__).resolve().parents[1].joinpath(".gitignore").read_text().splitlines()
    for rule in ("data/", "state/", "logs/", "reports/", "models/"):
        assert rule not in ignore, f"unanchored '{rule}' would also ignore src/quantbot/{rule}; use '/{rule}'"
