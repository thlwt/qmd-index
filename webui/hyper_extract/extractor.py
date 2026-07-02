"""
Entity and Relationship Extraction via LLM.
Uses the configured LLM API (settings.json) to extract structured
knowledge from document content.
"""

import json
import re
import urllib.request
import urllib.error


EXTRACT_SYSTEM_PROMPT = """You are a knowledge extraction engine. Analyze the given text and extract:
1. Entities: key concepts, people, organizations, locations, technologies, terms
2. Relationships: meaningful connections between entities

Return ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "entities": [
    {"name": "Entity Name", "type": "concept|person|organization|location|technology|term", "description": "brief description"}
  ],
  "relationships": [
    {"source": "Entity Name", "target": "Entity Name", "type": "relationship_type", "context": "sentence that shows the relationship"}
  ]
}

Rules:
- Extract 3-15 most important entities
- Only include relationships that are explicitly stated in the text
- Use lowercase-with-hyphens for relationship types (e.g., "works-for", "part-of", "located-in", "depends-on", "related-to", "creates", "uses", "implements", "extends")
- Each entity name should be concise (2-5 words max)
- Description should be 5-15 words"""


class HyperExtractor:
    """Extracts entities and relationships from document text using LLM."""

    def __init__(self, llm_url: str = "", llm_model: str = "",
                 llm_key: str = ""):
        self.llm_url = llm_url.rstrip("/")
        self.llm_model = llm_model
        self.llm_key = llm_key

    @classmethod
    def from_settings(cls, settings: dict):
        url = settings.get("llm_url", "http://127.0.0.1:5000/v1")
        return cls(
            llm_url=url,
            llm_model=settings.get("llm_model", ""),
            llm_key=settings.get("llm_key", ""),
        )

    def extract(self, text: str, max_chars: int = 6000) -> dict:
        """Extract entities and relationships from text."""
        if not text or not text.strip():
            return {"entities": [], "relationships": []}

        text = text[:max_chars]

        payload = json.dumps({
            "model": self.llm_model or "default",
            "messages": [
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract knowledge from:\n\n{text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
            "reasoning": {"enabled": False},
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.llm_key:
            headers["Authorization"] = f"Bearer {self.llm_key}"

        try:
            url = self.llm_url.rstrip("/") + "/chat/completions"
            req = urllib.request.Request(url, data=payload, headers=headers,
                                         method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
            msg = body.get("choices", [{}])[0].get("message", {})
            # Combine content + reasoning_content; search for JSON in the full output
            raw_full = (msg.get("content", "") or "") + "\n" + (msg.get("reasoning_content", "") or "")
            raw = raw_full.strip()
            return self._parse_response(raw)
        except Exception as e:
            # Try completion endpoint as fallback
            try:
                url2 = self.llm_url.rstrip("/") + "/completions"
                payload2 = json.dumps({
                    "model": self.llm_model or "default",
                    "prompt": f"{EXTRACT_SYSTEM_PROMPT}\n\nText:\n{text}\n\nJSON:",
                    "temperature": 0.1,
                    "max_tokens": 2048,
                }).encode("utf-8")
                req2 = urllib.request.Request(url2, data=payload2,
                                              headers=headers, method="POST")
                with urllib.request.urlopen(req2, timeout=300) as resp2:
                    body2 = json.loads(resp2.read())
                raw = (body2.get("choices", [{}])[0]
                       .get("text", ""))
                return self._parse_response(raw)
            except Exception:
                return {"entities": [], "relationships": [],
                        "error": str(e)}

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM JSON response, cleaning markdown fences if present."""
        if not raw:
            return {"entities": [], "relationships": []}
        raw = raw.strip()
        # Remove markdown code fences
        raw = re.sub(r'^```(?:json)?\s*\n?', '', raw)
        raw = re.sub(r'\n?```\s*$', '', raw)
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON block with balanced braces
            data = self._extract_json_block(raw)
        if data:
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            for e in entities:
                e.setdefault("type", "concept")
                e.setdefault("description", "")
            for r in relationships:
                r.setdefault("type", "related-to")
                r.setdefault("context", "")
            return {"entities": entities, "relationships": relationships}
        return {"entities": [], "relationships": [],
                "error": "No JSON found in response", "raw": raw[:500]}

    def _extract_json_block(self, text: str) -> dict | None:
        """Scan text for a JSON block with balanced braces."""
        # Find candidate JSON start positions
        for idx, ch in enumerate(text):
            if ch == '{':
                depth = 0
                in_str = False
                esc = False
                for j in range(idx, len(text)):
                    c = text[j]
                    if esc:
                        esc = False
                        continue
                    if c == '\\' and in_str:
                        esc = True
                        continue
                    if c == '"' and not esc:
                        in_str = not in_str
                        continue
                    if not in_str:
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                candidate = text[idx:j+1]
                                try:
                                    return json.loads(candidate)
                                except json.JSONDecodeError:
                                    break  # continue searching
                # If outer braces didn't close, reset search
        return None
