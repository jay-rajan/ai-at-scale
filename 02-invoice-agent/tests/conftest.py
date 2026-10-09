import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls the real model on OpenRouter (costs a few cents)")


def pytest_collection_modifyitems(config, items):
    import pytest
    if config.getoption("-m") and "live" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="live test: run with  pytest -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
