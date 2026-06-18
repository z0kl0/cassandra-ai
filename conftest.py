import os
import sys

import pytest

# Ensure the project root is importable so tests can `import forensics`/`curated_cases`
# regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", default=False,
                     help="run tests that hit the live SEC EDGAR API (network).")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires live SEC network access")


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.live tests unless --run-live is passed (keeps the default suite offline)."""
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live (network)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
