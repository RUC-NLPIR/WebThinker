"""Shared pytest fixtures and configuration for all tests."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file in the temp directory."""
    temp_path = temp_dir / "test_file.txt"
    temp_path.write_text("test content")
    yield temp_path


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "api_key": "test_api_key",
        "model": "test_model",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30,
        "base_url": "https://api.example.com",
        "debug": True,
    }


@pytest.fixture
def sample_json_data() -> Dict[str, Any]:
    """Provide sample JSON data for testing."""
    return {
        "id": "test_123",
        "name": "Test Item",
        "description": "A test item for unit testing",
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "tags": ["test", "sample", "fixture"],
        },
        "items": [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
            {"id": 3, "value": "third"},
        ],
    }


@pytest.fixture
def json_file(temp_dir: Path, sample_json_data: Dict[str, Any]) -> Path:
    """Create a temporary JSON file with sample data."""
    json_path = temp_dir / "test_data.json"
    with open(json_path, "w") as f:
        json.dump(sample_json_data, f, indent=2)
    return json_path


@pytest.fixture
def mock_api_client() -> Mock:
    """Create a mock API client for testing."""
    client = Mock()
    client.get = MagicMock(return_value={"status": "success", "data": []})
    client.post = MagicMock(return_value={"status": "created", "id": "new_123"})
    client.put = MagicMock(return_value={"status": "updated"})
    client.delete = MagicMock(return_value={"status": "deleted"})
    return client


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Provide a mock LLM response structure."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from the LLM.",
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }


@pytest.fixture
def mock_search_results() -> list[Dict[str, Any]]:
    """Provide mock search results for testing."""
    return [
        {
            "title": "First Result",
            "url": "https://example.com/1",
            "snippet": "This is the first search result",
            "score": 0.95,
        },
        {
            "title": "Second Result",
            "url": "https://example.com/2",
            "snippet": "This is the second search result",
            "score": 0.87,
        },
        {
            "title": "Third Result",
            "url": "https://example.com/3",
            "snippet": "This is the third search result",
            "score": 0.73,
        },
    ]


@pytest.fixture
def env_vars(monkeypatch) -> Dict[str, str]:
    """Set and clean up environment variables for testing."""
    test_vars = {
        "TEST_API_KEY": "test_key_12345",
        "TEST_ENV": "testing",
        "DEBUG": "true",
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_vars


@pytest.fixture
def mock_file_system(temp_dir: Path) -> Dict[str, Path]:
    """Create a mock file system structure for testing."""
    structure = {
        "data": temp_dir / "data",
        "outputs": temp_dir / "outputs",
        "logs": temp_dir / "logs",
        "cache": temp_dir / "cache",
    }
    
    for dir_path in structure.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create some sample files
    (structure["data"] / "sample.txt").write_text("Sample data content")
    (structure["outputs"] / "result.json").write_text('{"result": "success"}')
    
    return structure


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # This is a placeholder for resetting global state
    # Add any singleton reset logic here if needed
    yield


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog


@pytest.fixture
def benchmark_timer():
    """Simple timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Pytest configuration hooks

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: Mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: Mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)