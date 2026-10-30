"""Normalize short/long sections embedded in a JSON caption string."""

from __future__ import annotations

import re
from typing import Any


_CAPTION_HEADING = re.compile(
    r"^[ \t]*###[ \t]*(short|long)(?:[ \t]+description)?[ \t]*[:：][ \t]*",
    flags=re.IGNORECASE | re.MULTILINE,
)


def normalize_embedded_caption_sections(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    """Unwrap a heading-delimited description before selecting the caption mode.

    Only recognize an explicit caption envelope starting with a Short/Long
    heading. Leave normal prose, unrelated JSON fields and custom schemas alone.
    Explicit non-empty caption fields take precedence over embedded sections.
    """
    result = dict(payload)
    description = result.get("description")
    if not isinstance(description, str):
        return result

    headings = list(_CAPTION_HEADING.finditer(description))
    if not headings or description[: headings[0].start()].strip():
        return result
    names = [match.group(1).lower() for match in headings]
    if len(names) != len(set(names)):
        return result

    sections = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(description)
        text = description[heading.end() : end].strip()
        if text:
            sections[f"{names[index]}_description"] = text
    if not sections:
        return result

    result.pop("description")
    for alias in ("short", "long"):
        key = f"{alias}_description"
        explicit = result.pop(alias, None)
        if not result.get(key) and explicit:
            result[key] = explicit
    for key, text in sections.items():
        if not result.get(key):
            result[key] = text
    if mode == "long":
        result.pop("short", None)
        result.pop("short_description", None)
    elif mode == "short":
        result.pop("long", None)
        result.pop("long_description", None)
    return result
