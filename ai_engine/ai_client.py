import os
import json
import logging
from typing import Any, Dict, List

import openai  # type: ignore

logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are an expert quantitative developer and Python performance engineer.
You are helping improve an algorithmic crypto trading bot in production.

STRICT OUTPUT FORMAT:
- Reply with ONLY valid JSON.
- JSON schema:
{
  "analysis": "string",
  "risks": ["string", ...],
  "recommended_changes": ["string", ...],
  "updated_files": [
    {
      "path": "relative/path/from_bot_root.py",
      "reason": "what this change does",
      "content": "FULL NEW CONTENT of the file"
    }
  ]
}

Rules:
- Focus on robustness, risk controls, and performance.
- Never increase leverage without tightening risk controls.
- Prefer small, incremental, safe improvements.
- Only include up to 3 updated_files per run.
- If unsure about a file, do NOT include it.
"""

def request_improvements(context: str) -> Dict[str, Any]:
    if not openai.api_key:
        logger.warning("OPENAI_API_KEY not set; AI engine disabled.")
        return {
            "analysis": "OPENAI_API_KEY not set; AI review skipped.",
            "risks": [],
            "recommended_changes": [],
            "updated_files": [],
        }

    model = os.getenv("AI_ENGINE_MODEL", "gpt-4.1-mini")

    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("AI response not valid JSON, saving raw content.")
        parsed = {
            "analysis": "AI response was not valid JSON.",
            "risks": [],
            "recommended_changes": [],
            "updated_files": [],
            "raw": raw,
        }
    return parsed
