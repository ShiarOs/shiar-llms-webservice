"""Bad environment values must fail before any model or cache request."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["llm_app.config", "solution.llm_app.config"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("OLLAMA_TIMEOUT", "0"),
        ("OLLAMA_TIMEOUT", "nan"),
        ("OLLAMA_TIMEOUT", "inf"),
        ("CACHE_TTL", "0"),
        ("CACHE_TTL", "-1"),
        ("MIN_RELEVANCE", "nan"),
        ("MIN_RELEVANCE", "1.1"),
    ],
)
def test_invalid_configuration_is_explicit(module, name, value):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env={**os.environ, name: value},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"ValueError: {name}" in result.stderr
