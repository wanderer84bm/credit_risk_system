from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()  # reads .env into environment
import math, json, joblib, numpy as np, pandas as pd, warnings
import os, requests
from datetime import date, datetime
import math
import numbers
import json

from ast import literal_eval

import codes as rc

def _norm(s): return (s or "").strip().lower()

def _map_ml_flags_to_codes(ml_flags):
    codes = []
    for f in ml_flags or []:
        code = rc.ML_FLAG_TO_CODE.get((f or "").strip().upper(), "")
        if code: rc.append(code)
    return codes

def _map_det_flags_to_codes(det_flags):
    """
    det_flags is list of dicts like:
      {"flag_type":"red","name":"Housing cost burden (severe)","reasoning":"...", "severity":"high"}
    """
    codes = []
    for f in det_flags or []:
        name = _norm(f.get("name") if isinstance(f, dict) else f)
        code = rc.DET_NAME_TO_CODE.get(name, "")
        if not code:
            # light substring fallback (v0)
            if "housing cost burden" in name or "residual" in name: code = "R006"
            elif "balance exceeds limit" in name or "over limit" in name: code = "R003"
            elif "utilization" in name and "inconsistent" in name: code = "R013"
            elif "velocity" in name or "tradeline" in name or "inquiries" in name: code = "R002"
            elif "history" in name and "age" in name: code = "R013"
            elif "residence" in name or "housing cost" in name: code = "R013"
        if code: codes.append(code)
    return codes

def _dedupe_keep_order(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def _sort_by_priority(codes):
    rank = {c:i for i, c in enumerate(rc.PRIORITY)}
    return sorted(codes, key=lambda c: rank.get(c, 999))

def agent6(ml_flags, det_flags, agent3_passed: bool, limit: int = 3):
    """
    - If agent3_passed == False  -> policy decline: AA013 only
    - Else                       -> map flags to codes, return up to `limit` codes by priority
    Returns:
      {"basis": "policy"|"flags",
       "codes": [...], "labels": [...],
       "trace": {"ml_flags": [...], "det_flags": [...], "ml_mapped": [...], "det_mapped": [...]}}
    """
    # Agent 3 gate: policy decline if it failed upstream
    if not bool(agent3_passed):
        return {
            "basis": "policy",
            "codes": ["P001"],
            "labels": [rc.CODES["P001"]],
            "trace": {"ml_flags": ml_flags or [], "det_flags": det_flags or [], "ml_mapped": [], "det_mapped": ["P001 (agent3_failed)"]}
        }

    # Otherwise map flags → codes
    det_codes = _map_det_flags_to_codes(det_flags)
    ml_codes  = _map_ml_flags_to_codes(ml_flags)

    merged = _dedupe_keep_order(det_codes + ml_codes)
    merged = _sort_by_priority(merged)
    if isinstance(limit, int) and limit > 0:
        merged = merged[:limit]

    return {
        "basis": "flags",
        "codes": merged,
        "labels": [rc.CODES[c] for c in merged],
        "trace": {"ml_flags": ml_flags or [], "det_flags": det_flags or [], "ml_mapped": ml_codes, "det_mapped": det_codes}
    }
