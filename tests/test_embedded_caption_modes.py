"""Regression coverage for JSON-wrapped ###Short / ###Long captions."""

import copy
import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from module.caption_pipeline.postprocess import postprocess_caption_content
from module.providers.base import CaptionResult, ProviderContext
from module.providers.cloud_vlm.openai_compatible import OpenAICompatibleProvider
from utils.output_writer import write_caption_output


SHORT = "A pink ornamental bottle."
LONG = "A glossy pink bottle.\nIts stopper resembles a flower bud: overlapping petals are visible."
WRAPPED = f"###Short: {SHORT}\n\n###Long: {LONG}"
METADATA = {"tags": ["bottle"], "scores": {"quality": 8}, "confidence": 0.9}


def _args(mode="long"):
    return SimpleNamespace(mode=mode, pair_dir="", image_prompt_template="")


def _provider():
    return OpenAICompatibleProvider(ProviderContext(console=Console(file=io.StringIO()), args=_args(), config={}))


def _expected(mode):
    fields = {"short_description": SHORT, "long_description": LONG}
    if mode == "long":
        fields.pop("short_description")
    elif mode == "short":
        fields.pop("long_description")
    return fields


@pytest.mark.parametrize("mode", ["long", "short", "all"])
def test_openai_filter_unwraps_description_before_selecting_mode(mode):
    payload = {"description": WRAPPED, **METADATA}
    before = copy.deepcopy(payload)

    actual = _provider()._filter_by_mode(payload, mode)

    assert actual == {**_expected(mode), **METADATA}
    assert payload == before


@pytest.mark.parametrize("mode", ["long", "short", "all"])
@pytest.mark.parametrize("representation", ["dict", "json", "parsed_result", "raw_result", "fenced_json"])
def test_postprocess_and_publication_filter_embedded_sections(tmp_path, mode, representation):
    payload = {"description": WRAPPED, **METADATA}
    before = copy.deepcopy(payload)
    output = payload
    if representation == "json":
        output = json.dumps(payload)
    elif representation == "parsed_result":
        output = CaptionResult(raw=json.dumps(payload), parsed=payload, metadata={"provider": "offline"})
    elif representation == "raw_result":
        output = CaptionResult(raw=json.dumps(payload), metadata={"provider": "offline"})
    elif representation == "fenced_json":
        output = "```json\n" + json.dumps(payload) + "\n```"

    args = _args(mode)
    if representation == "fenced_json":
        args.image_prompt_template = "danbooru_tags"
    image = tmp_path / "bottle.png"
    result = postprocess_caption_content(output, image, args, Console(file=io.StringIO()))
    expected = {**_expected(mode), **METADATA}

    assert result.parsed == expected
    assert json.loads(result.raw) == expected
    assert result.is_persistable
    assert payload == before
    if representation in {"parsed_result", "raw_result"}:
        assert result.metadata == {"provider": "offline"}
    txt_path, json_path = write_caption_output(image, result, "image/png")
    assert txt_path.read_text(encoding="utf-8") == (SHORT if mode == "short" else LONG)
    assert json.loads(json_path.read_text(encoding="utf-8")) == expected
    assert json.loads(result.to_dataset_caption()) == expected


@pytest.mark.parametrize("wrapped", [
    f"### Short: {SHORT}\r\n\r\n### Long: {LONG}",
    f"###SHORT DESCRIPTION： {SHORT}\n###long description: {LONG}",
    f"###Long: {LONG}\n###Short: {SHORT}",
])
def test_caption_heading_variants_preserve_body_and_punctuation(wrapped):
    assert _provider()._filter_by_mode({"description": wrapped}, "long") == {"long_description": LONG}


@pytest.mark.parametrize("mode,key,text", [("long", "long_description", LONG), ("short", "short_description", SHORT)])
def test_single_caption_heading_is_removed(mode, key, text):
    assert _provider()._filter_by_mode({"description": f"###{mode.title()}: {text}"}, mode) == {key: text}


def test_explicit_description_fields_take_precedence():
    payload = {"description": WRAPPED, "long_description": "Explicit caption.", "short_description": "Explicit short."}
    assert _provider()._filter_by_mode(payload, "long") == {"long_description": "Explicit caption."}


@pytest.mark.parametrize("mode", ["long", "short", "all"])
@pytest.mark.parametrize("alias", ["long", "short"])
def test_explicit_aliases_are_canonicalized_before_publication(tmp_path, mode, alias):
    payload = {"description": WRAPPED, alias: "Explicit caption.", **METADATA}
    before = copy.deepcopy(payload)
    expected = {**_expected(mode), **METADATA}
    key = f"{alias}_description"
    if key in expected:
        expected[key] = "Explicit caption."
    result = postprocess_caption_content(payload, tmp_path / "bottle.png", _args(mode), Console(file=io.StringIO()))
    assert result.parsed == expected
    assert result.is_persistable
    assert payload == before
    text_path, json_path = write_caption_output(tmp_path / "bottle.png", result, "image/png")
    assert text_path.read_text(encoding="utf-8") == expected.get("long_description", expected.get("short_description"))
    assert json.loads(json_path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize("description", [
    "A bottle: pink glass and a flower stopper.",
    "A sign reads ###Short: and ###Long: in printed lettering.",
    "### Details: A bottle.",
    "", None, {"text": WRAPPED},
])
def test_unrelated_descriptions_are_not_rewritten(description):
    payload = {"description": description, **METADATA}
    assert _provider()._filter_by_mode(payload, "long") == payload


def test_custom_json_structure_is_not_recursively_rewritten():
    payload = {"high_level_description": "A bottle.", "elements": [{"description": WRAPPED}]}
    result = postprocess_caption_content(payload, "image.png", _args(), Console(file=io.StringIO()))
    assert result.parsed == payload


def test_non_image_structured_payload_is_not_rewritten():
    payload = {"description": WRAPPED, "transcript": "Preserve this transcript."}
    result = postprocess_caption_content(payload, "audio.wav", _args(), Console(file=io.StringIO()))
    assert result.parsed == payload
