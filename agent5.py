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
import cutoff as c

APPROVED = "APPROVE"
APPROVED_WITH_CONDITION = "APPROVE_WITH_CONDITION"
REJECT = "REJECT"

def agent5(out_from_4, score, profile, outth):
    if outth == False:
        return REJECT
    else:
        ml = out_from_4["risk_score"]
        #pd = out_from_4["pd"]
        final_score =  0.8*ml + 0.2*score
        
        if final_score < c.APPROVE_SCORE:
            return APPROVED 
        elif final_score > c.REJECT_SCORE:
            return REJECT 
        else: 
            return APPROVED_WITH_CONDITION 