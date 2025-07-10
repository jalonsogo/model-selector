"""
Pytest configuration and fixtures
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_home_dir(temp_dir, monkeypatch):
    """Mock home directory for testing"""
    monkeypatch.setattr(Path, 'home', lambda: temp_dir)
    return temp_dir


@pytest.fixture(autouse=True)
def cleanup_imports():
    """Clean up module imports after each test"""
    import sys
    modules_to_cleanup = [
        mod for mod in sys.modules.keys() 
        if mod.startswith('model_selector.')
    ]
    
    yield
    
    for module in modules_to_cleanup:
        if module in sys.modules:
            del sys.modules[module]