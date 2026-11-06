# Testing Service Boundaries

In lessons 01 and 02 you checked the service by hand, with Ollama and Redis
running. In this lesson you look at the supplied unit tests that check the same
behavior automatically. They establish that this application validates
requests, delegates correctly, handles cache outcomes, and translates failures.

Ollama and Redis are separate services with their own behavior, and testing
them is not this application's job, so the tests replace them with test doubles
(mocks). As a result,
the tests run in under a second, give the same result every time, and can run
anywhere, including a CI pipeline.

## Route Tests

[HTTPX's ASGI transport](https://www.python-httpx.org/advanced/transports/#asgi-transport)
sends in-process requests to the FastAPI application without starting a server.

The route has one dependency: `send_prompt`, the function it calls to get an
answer. **Patching** replaces a dependency with a mock for the duration of one
test, using [`unittest.mock.patch`](https://docs.python.org/3/library/unittest.mock.html#patch).
The mock returns a value you choose and records how it was called, and the
original is restored when the test ends.

Patch the dependency where the route
imports it. This test from
[`llm_app/tests/test_generate.py`](llm_app/tests/test_generate.py) replaces
`send_prompt` with a mock that returns a fixed response:

```python
MOCK_RESPONSE = GenerateResponse(model="qwen3.5:2b", response="Mocked output")


async def post(path: str, payload: dict[str, object]) -> Response:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(path, json=payload)


@patch(
    "llm_app.fastapi_llms.send_prompt",
    new_callable=AsyncMock,
    return_value=MOCK_RESPONSE,
)
def test_generate_returns_typed_response(mock_send: AsyncMock) -> None:
    response = asyncio.run(post("/generate", {"prompt": "Hello"}))

    assert response.status_code == 200
    assert response.json() == {
        "model": "qwen3.5:2b",
        "response": "Mocked output",
    }
    call = mock_send.await_args
    assert call is not None
    request = call.args[0]
    assert request.model == "qwen3.5:2b"
```

The test functions are ordinary synchronous functions, so they use
`asyncio.run(...)` to run the async `post` helper to completion.

Why patch `llm_app.fastapi_llms.send_prompt` and not
`llm_app.utils.send_prompt`, where the function is defined? The line
`from llm_app.utils import send_prompt` in `fastapi_llms.py` copies the name
into that module, and the route looks it up there. Patching
`llm_app.utils.send_prompt` would leave the already-imported route reference
unchanged. Patch the name looked up by the code under test (see
[Where to patch](https://docs.python.org/3/library/unittest.mock.html#where-to-patch)).

## Utility Tests

The cache and model client are separate boundaries. The supplied tests in
[`llm_app/tests/test_utils.py`](llm_app/tests/test_utils.py) cover these cases:

- a cache hit returns without constructing or invoking Ollama,
- a cache miss invokes Ollama and writes the result to the cache,
- a Redis read failure still invokes Ollama,
- an Ollama failure becomes an HTTP `502` response,
- an Ollama timeout becomes an HTTP `504` response,
- changing any generation parameter changes the cache key.

Each test replaces Redis and Ollama separately, so it can control one boundary
while checking the other. For example, the cache-hit test makes Redis return a
stored answer and checks that the Ollama client is never created:

```python
def test_cache_hit_skips_ollama() -> None:
    payload = GenerateRequest(prompt="Hello")
    with (
        patch("llm_app.utils.redis_client.get", new=AsyncMock(return_value="Cached")),
        patch("llm_app.utils.OllamaLLM") as llm_class,
    ):
        response = asyncio.run(send_prompt(payload))

    assert response.response == "Cached"
    llm_class.assert_not_called()
```

You will use the same pattern in lesson 04 to test the RAG helper.

## Run The Tests

Run the supplied application tests:

```bash
uv run pytest llm_app/tests -v
```

All tests should pass in under a second. They run in-process, so the API from
lesson 01 can keep running in its own terminal.

The root [`conftest.py`](conftest.py) prevents test collection from loading
`.env` and fixes test configuration before application imports. This matters
because `llm_app/config.py` reads the environment, and `llm_app/utils.py`
creates the Redis client, as soon as they are imported. `conftest.py` points
Ollama and Redis at an unreachable address (`127.0.0.1:1`) first, so a call
someone forgot to mock fails fast instead of quietly reaching a real service.
Model and cache calls remain mocked; changing your live service settings must
not change unit test behavior.

## Check Your Understanding

1. `test_generate_returns_typed_response` passes with status `200`. Why is that
   not evidence that caching works, and which tests do check caching?
2. What would happen if the route test patched `llm_app.utils.send_prompt`
   instead of `llm_app.fastapi_llms.send_prompt`?
3. A new test calls `send_prompt` but forgets to patch `OllamaLLM`. What happens
   when it runs, and why is that better than the alternative?

<details>
<summary>Show Reference Answers</summary>

1. The route test replaces the whole `send_prompt` function with a mock, so
   none of the cache code runs. It only shows that the route validates the
   request, delegates to `send_prompt`, and serializes the response. Caching is
   checked by the tests in `llm_app/tests/test_utils.py`, which call the real
   `send_prompt` and patch `redis_client.get`, `redis_client.set`, and
   `OllamaLLM` separately (for example `test_cache_hit_skips_ollama` and
   `test_cache_miss_generates_and_stores_response`).
2. The route would still call the real `send_prompt`, because
   `fastapi_llms.py` imported its own reference to the function and the route
   looks the name up there. The real function would try Redis and Ollama at the
   unreachable addresses set by `conftest.py`, return `502`, and the test's
   `200` assertion would fail.
3. `conftest.py` points `OLLAMA_BASE_URL` at `127.0.0.1:1`, where nothing is
   listening, so the call fails immediately and `send_prompt` raises a `502`.
   The test fails fast with a clear error instead of quietly calling a real
   Ollama model, which would make it slow and dependent on what is running on
   the developer's machine.

</details>
