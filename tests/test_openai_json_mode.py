"""JSON-mode message validation using the real SDK and an offline HTTP endpoint."""

import copy
import io
import json

import openai
import pytest
from rich.console import Console

try:
    import httpx2 as httpx
except ImportError:
    import httpx

from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
from module.providers.cloud_vlm.openai_compatible import OpenAICompatibleProvider, _ensure_json_instruction
from tests.provider_v2_helpers import make_provider_args


JSON_VALIDATION_ERROR = (
    "Response input messages must contain the word 'json' in some form "
    "to use 'text.format' of type 'json_object'."
)


def _message_text(messages):
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    yield part.get("text", "")


def _completion(text):
    return {
        "id": "offline-json-test",
        "object": "chat.completion",
        "created": 0,
        "model": "offline-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
    }


@pytest.fixture
def endpoint(monkeypatch):
    requests = []
    replies = []
    original_client = openai.OpenAI

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("response_format") == {"type": "json_object"} and not any(
            "json" in text.lower() for text in _message_text(body["messages"])
        ):
            return httpx.Response(400, json={"error": {"message": JSON_VALIDATION_ERROR, "type": "invalid_request_error"}})
        status, payload = replies.pop(0) if replies else (200, _completion('{"description":"A red fruit icon."}'))
        return httpx.Response(status, json=payload)

    def factory(**kwargs):
        return original_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setattr("module.providers.utils.time.sleep", lambda _seconds: None)
    return requests, replies


def _provider(**overrides):
    args = make_provider_args(
        openai_api_key="offline-placeholder",
        openai_base_url="https://offline.invalid/v1",
        openai_model_name="offline-model",
        openai_json_mode=True,
        mode="all",
        max_retries=3,
        wait_time=0,
    )
    vars(args).update(overrides)
    return OpenAICompatibleProvider(ProviderContext(console=Console(file=io.StringIO()), args=args, config={}))


def _media(**overrides):
    args = dict(uri="offline.png", mime="image/png", sha256hash="offline", modality=MediaModality.IMAGE, blob="aW1hZ2U=")
    args.update(overrides)
    return MediaContext(**args)


@pytest.mark.parametrize("system", ["", "Describe only visible details."])
def test_json_mode_without_json_in_prompt_passes_server_validation(endpoint, system):
    requests, _ = endpoint
    provider = _provider()
    prompts = PromptContext(system=system, user="描述这个水果图标。")
    original = provider._build_messages(_media(), prompts)

    result = provider.attempt(_media(), prompts)

    assert result.description == "A red fruit icon."
    assert result.is_persistable
    assert len(requests) == 1
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "json" in requests[0]["messages"][0]["content"].lower()
    assert requests[0]["messages"][0]["content"].startswith(system)
    assert requests[0]["messages"][1:] == original[1:]
    assert prompts.system == system


@pytest.mark.parametrize(
    ("system", "user"),
    [("Return JSON with the specified schema.", "Describe it."), ("Describe it.", 'Return a JsOn object with "text".')],
)
def test_existing_json_schema_and_instructions_are_unchanged(endpoint, system, user):
    requests, _ = endpoint
    provider = _provider()
    prompts = PromptContext(system=system, user=user)
    expected = provider._build_messages(_media(), prompts)

    assert provider.attempt(_media(), prompts).is_persistable

    assert requests[0]["messages"] == expected


def test_json_in_image_data_does_not_count_as_a_text_instruction(endpoint):
    requests, _ = endpoint
    provider = _provider()
    media = _media(blob="jsonIsNotAnInstruction")

    assert provider.attempt(media, PromptContext(system="Describe it.", user="Describe the icon.")).is_persistable
    assert "json" in requests[0]["messages"][0]["content"].lower()


def test_disabled_json_mode_preserves_original_messages(endpoint):
    requests, replies = endpoint
    replies.append((200, _completion("A red fruit icon.")))
    provider = _provider(openai_json_mode=False)
    prompts = PromptContext(system="Return a plain caption.", user="Describe it.")

    result = provider.attempt(_media(), prompts)

    assert result.raw == "A red fruit icon."
    assert "response_format" not in requests[0]
    assert requests[0]["messages"] == provider._build_messages(_media(), prompts)


def test_unsupported_format_fallback_removes_injected_json_instruction(endpoint):
    requests, replies = endpoint
    replies.extend([
        (400, {"error": {"message": "response_format json_object is not supported", "type": "invalid_request_error"}}),
        (200, _completion("A red fruit icon.")),
    ])
    provider = _provider()
    prompts = PromptContext(system="Describe the image.", user="Return the caption.")

    result = provider.attempt(_media(), prompts)

    assert result.raw == "A red fruit icon."
    assert len(requests) == 2
    assert "response_format" not in requests[1]
    assert requests[1]["messages"] == provider._build_messages(_media(), prompts)


def test_502_retry_keeps_json_format_and_valid_message_text(endpoint, monkeypatch):
    requests, replies = endpoint
    replies.extend([
        (502, {"error": {"message": "Upstream service temporarily unavailable", "type": "upstream_error"}}),
        (200, _completion('{"description":"A red fruit icon."}')),
    ])
    provider = _provider()
    monkeypatch.setattr(provider, "prepare_media", lambda *_args: _media())
    monkeypatch.setattr(provider, "resolve_prompts", lambda *_args, **_kwargs: PromptContext(system="Describe.", user="Caption."))

    assert provider.execute("offline.png", "image/png", "offline").is_persistable
    assert len(requests) == 2
    assert all(body["response_format"] == {"type": "json_object"} for body in requests)
    assert all(any("json" in text.lower() for text in _message_text(body["messages"])) for body in requests)


def test_video_json_mode_preserves_video_part(endpoint, tmp_path):
    requests, _ = endpoint
    video = tmp_path / "offline.mp4"
    video.write_bytes(b"offline video fixture")
    media = _media(uri=str(video), mime="video/mp4", modality=MediaModality.VIDEO, blob=None)
    provider = _provider()
    prompts = PromptContext(system="Describe the video.", user="Describe visible events.")
    expected_user = copy.deepcopy(provider._build_messages(media, prompts)[1])

    assert provider.attempt(media, prompts).is_persistable
    assert requests[0]["messages"][1] == expected_user
    assert "json" in requests[0]["messages"][0]["content"].lower()


def test_pair_json_mode_preserves_both_images(endpoint):
    requests, _ = endpoint
    provider = _provider(pair_dir="pairs")
    media = _media(pair_blob="cGFpcg==")
    prompts = PromptContext(system="Compare the images.", user="Describe the difference.")
    expected_user = copy.deepcopy(provider._build_messages(media, prompts)[1])

    assert provider.attempt(media, prompts).is_persistable
    assert requests[0]["messages"][1] == expected_user
    assert sum(part["type"] == "image_url" for part in expected_user["content"]) == 2


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": "Describe."}, {"role": "user", "content": "Caption."}],
    [{"role": "user", "content": "描述水果图标"}],
    [{"role": "system", "content": [{"type": "text", "text": "Describe."}]}],
])
def test_json_instruction_never_mutates_original_messages(messages):
    before = copy.deepcopy(messages)

    updated = _ensure_json_instruction(messages)

    assert any("json" in text.lower() for text in _message_text(updated))
    assert messages == before
