import json
from typing import Optional

from .config import OPENAI_API_KEY, OPENAI_MODEL


def call_llm_json(system_prompt: str, user_prompt: str) -> Optional[dict]:
    """Call OpenAI only when the package and API key are available."""
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        return None
    return json.loads(content)
