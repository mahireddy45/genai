from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class AssistantOutput(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = []


def validate_assistant_output(raw_text: str) -> AssistantOutput:
    """Attempt to validate/parse assistant output into structured form.

    This function first tries to parse JSON embedded in the assistant text.
    If parsing fails, it will wrap the raw text as the `answer` field.
    """
    import json

    # try to find a JSON object in the text
    text = raw_text.strip()
    try:
        # If the assistant returned just JSON, parse it directly
        data = json.loads(text)
        return AssistantOutput.parse_obj(data)
    except Exception:
        # try to find a JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end+1])
                return AssistantOutput.parse_obj(data)
            except Exception:
                pass

    # Fallback: return raw text as answer
    try:
        return AssistantOutput(answer=text, sources=[])
    except ValidationError as e:
        logger.exception("Assistant output failed validation: %s", e)
        # As a last resort, return minimal structure
        return AssistantOutput(answer=text)
