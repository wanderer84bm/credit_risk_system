import os, requests, json, uuid, re
from typing import Any, Dict, List, Union

def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

def _extract_json_from_text(text: str) -> Union[dict, list, None]:
    """
    Try to pull the first JSON object/array from arbitrary text, incl. code fences.
    """
    if not isinstance(text, str):
        return None

    # 1) strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    if fenced:
        blob = fenced.group(1)
        try:
            return json.loads(blob)
        except Exception:
            pass

    # 2) greedy search for a {...} or [...] block
    for opener, closer in [("{", "}"), ("[", "]")]:
        try:
            start = text.index(opener)
            end = text.rindex(closer)
            candidate = text[start:end+1]
            if _looks_like_json(candidate):
                return json.loads(candidate)
        except Exception:
            continue

    # 3) last resort: try raw
    try:
        return json.loads(text) if _looks_like_json(text) else None
    except Exception:
        return None

def _normalize_to_flags(obj: Any) -> Dict[str, Any]:
    """
    Accept many shapes and return {"flags":[...]} or {"flags":[]}
    """
    # If obj is already {"flags":[...]}
    if isinstance(obj, dict) and isinstance(obj.get("flags"), list):
        return {"flags": obj["flags"]}

    # If obj is a bare list (hopefully of flags)
    if isinstance(obj, list):
        return {"flags": obj}

    # If obj is a dict with some text field containing JSON (common Lyzr shape)
    if isinstance(obj, dict):
        # Try common keys where the agent’s text might live
        for key in ("agent_response", "response", "message", "output", "assistant_message"):
            val = obj.get(key)
            if isinstance(val, (dict, list)):
                # Maybe it already is JSON
                norm = _normalize_to_flags(val)
                if norm["flags"]:
                    return norm
            elif isinstance(val, str):
                parsed = _extract_json_from_text(val)
                if parsed is not None:
                    return _normalize_to_flags(parsed)

        # Sometimes responses are nested, so scan all string fields
        for v in obj.values():
            if isinstance(v, str):
                parsed = _extract_json_from_text(v)
                if parsed is not None:
                    return _normalize_to_flags(parsed)

    return {"flags": []}

def call_lyzr(profile: Dict[str, Any],
              initial_red_flags: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    POSTs to Lyzr chat and robustly parses whatever comes back,
    normalizing to {"flags":[...]} so you can score it directly.
    """
    url = os.getenv("LYZR_AGENT_URL") or "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
    api_key = os.getenv("LYZR_API_KEY")
    agent_id = os.getenv("LYZR_AGENT_ID")

    if not api_key or not agent_id:
        print("[lyzr] missing LYZR_API_KEY or LYZR_AGENT_ID")
        return {"flags": []}

    # Ensure flags is a LIST (you previously passed a dict by mistake)
    if isinstance(initial_red_flags, dict) and "flags" in initial_red_flags:
        initial_red_flags = initial_red_flags["flags"]

    payload_for_agent = {
        "profile": profile,
        "initial_red_flags": initial_red_flags,
        "remaining_slots": max(0, 5 - len(initial_red_flags)),
    }

    body = {
        "user_id": "default_user",
        "agent_id": agent_id,
        "session_id": f"{agent_id}-{uuid.uuid4().hex[:8]}",
        # Your agent prompt should instruct: “Return ONLY JSON with a 'flags' array…”
        "message": json.dumps(payload_for_agent),
    }
    headers = {"Content-Type": "application/json", "x-api-key": api_key}

    resp = requests.post(url, json=body, headers=headers, timeout=30)
    # If you ever hit 4xx/5xx again, print the body for debugging
    if not (200 <= resp.status_code < 300):
        print(f"[lyzr] HTTP {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()

    # Try JSON first; fall back to text
    try:
        data = resp.json()
    except Exception:
        data = resp.text

    # Robust normalization
    normalized = _normalize_to_flags(data)

    # If empty, show a short debug snippet so you can see what came back
    if not normalized["flags"]:
        preview = (resp.text or "")[:400]
        print("[lyzr] parsed no flags; response preview:\n", preview)

    return normalized


