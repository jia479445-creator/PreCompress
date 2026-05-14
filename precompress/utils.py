"""Lifted from `src/lightmem/memory/utils.py` (only `clean_response`)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def clean_response(response: str) -> List[Dict[str, Any]]:
    """
    Cleans the model response by:
    1. Removing enclosing code block markers (```[language] ... ```).
    2. Parsing the JSON content safely.
    3. Returning the value of the "data" key if present, otherwise trying to return the parsed list/dict.
    """
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response.strip())
    cleaned = match.group(1).strip() if match else response.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
        return []

    if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        return parsed["data"]

    if isinstance(parsed, list):
        return parsed

    return []
