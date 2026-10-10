"""Exercise the real exploration loops with offline search and model responses."""

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("module_name", ["run_web_thinker", "run_web_thinker_report"])
@pytest.mark.parametrize("action", ["search", "click", "limit"])
def test_exploration_does_not_repeat_previous_turns(monkeypatch, module_name, action):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--api_base_url",
            "http://unused",
            "--aux_api_base_url",
            "http://unused",
        ],
    )
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda _: SimpleNamespace(eos_token="[eos]")
    )
    runner = importlib.import_module(module_name)
    monkeypatch.setattr(
        runner, "tokenizer", SimpleNamespace(eos_token="[eos]"), raising=False
    )
    prompts = []
    main_calls = 0
    responses = []
    if action == "click":
        responses = [
            f"step-{i}:{runner.BEGIN_CLICK_LINK}https://example.com/page{runner.END_CLICK_LINK}"
            for i in range(2)
        ]
    else:
        responses = [
            f"step-{i}:{runner.BEGIN_SEARCH_QUERY}a specific query{runner.END_SEARCH_QUERY}"
            for i in range(11 if action == "limit" else 2)
        ]

    async def generate(**kwargs):
        nonlocal main_calls
        if "stop" in kwargs:
            prompts.append(kwargs["prompt"])
            response = (
                responses[main_calls] if main_calls < len(responses) else "Finished."
            )
            main_calls += 1
            return "INITIAL_CONTEXT", response
        if kwargs.get("generate_mode") == "completion":
            prompts.append(kwargs["prompt"])
            return "", "Final summary."
        return "", "page summary or click intent"

    monkeypatch.setattr(runner, "generate_response", generate)
    search = AsyncMock(return_value={})
    monkeypatch.setattr(runner, "google_serper_search_async", search)
    monkeypatch.setattr(runner, "extract_relevant_info_serper", lambda _: [])
    monkeypatch.setattr(runner, "format_search_results", lambda _: "SEARCH_EVIDENCE")
    monkeypatch.setattr(
        runner,
        "fetch_page_content_async",
        AsyncMock(return_value={"https://example.com/page": "Useful page content."}),
    )
    args = SimpleNamespace(
        model_name="test",
        aux_model_name="test",
        temperature=0.5,
        top_p=1,
        max_tokens=1000,
        repetition_penalty=1,
        top_k_sampling=20,
        min_p=0,
        search_engine="serper",
        serper_api_key="unused",
        top_k=3,
        use_jina=False,
        jina_api_key=None,
        keep_links=True,
    )
    kwargs = {
        "client": None,
        "aux_client": None,
        "search_query": "initial question",
        "document": "initial evidence",
        "search_intent": "find evidence",
        "args": args,
        "search_cache": {},
        "url_cache": {},
        "semaphore": None,
    }
    if module_name.endswith("report"):
        kwargs["question"] = "question"

    output, original_prompt = asyncio.run(runner.generate_deep_web_explorer(**kwargs))

    assert original_prompt == "INITIAL_CONTEXT"
    assert len(prompts) >= 3
    assert prompts[-1].count("step-0:") == 1
    assert prompts[-1].count("step-1:") == 1
    assert output.count("step-0:") == 1
    if action != "click":
        assert prompts[-1].count("SEARCH_EVIDENCE") == 1
        search.assert_awaited_once()
    if action == "limit":
        assert "Final summary." in output
