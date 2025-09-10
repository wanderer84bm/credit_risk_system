
# BNPL Credit Risk — agentic pipeline

**Goal:** build an agentic BNPL risk pipeline and optimize **Balanced Accuracy** (BA) end-to-end on a 1k stratified sample. **Current BA > 0.779.**

**Out of scope (for this repo/run):**
- No per-row Lyzr/LLM calls (speed/$)
- No third-party input verification
- No LLM-generated synthetic data

---

## Pipeline (at a glance)

```

1 Intake  →  2 Features  →  3 Gate  →  3.5 Flags  →  4 Model  →  5 Decision  →  6 Explain

```

### What each agent does
1. **Intake** — validate required fields; raise if missing.  
2. **Features** — derive **DTI, PTI, HCR, Residual Monthly Income, Credit Velocity**.  
3. **Gate (policy)** — hard policy checks; fail = auto-reject.  
   - **3.5 Flags** — red/green flags (LLM design), but **LLM calls are disabled in eval**.  
4. **Model (risk)** — HistGradientBoosting trained on **John Hull / Lending Club**; outputs PD and a **0–100 risk** score.  
5. **Decision** — blend risk + flags and compare to cutoff.  
6. **Explain** — map reasons to 6 short codes (and plain-English labels). Explanations come from **Agent 3 flags** + **top SHAP factors** from Agent 4.

---

## Decision policy

- **Score space:** risk **0–100**
- **Blend:**
```

final_score = 0.8 * ML_risk + 0.2 * flag_points

```
- **Flags:** max 5 per profile  
  - Red: **+12** (high), **+10** (med), **+8** (low)  
  - Green: **−15** (high), **−10** (med), **−5** (low)
- In this eval, **LLM flags are off**.
- **Cutoff & rule:** `final_score ≥ 23 → REJECT`, `final_score > 30 → REJECT`.
- **Binary labels:** APPROVE = **0**, REJECT = **1**.
- **Model note:** the model computes **probability of default (PD)** and it’s scaled ×100 to get the **risk score**.

---

## Metrics & target

- **Primary:** Balanced Accuracy = (TPR + TNR) / 2  
- **Secondary:** Accuracy, Precision (class=1), Recall (class=1), F1 (class=1)  
- **Class mapping:** 1 = default/bad, 0 = non-default/good  
- **Data balance (eval slice):** ~80% goods / 20% bads (1k rows)

---

## Current results (1k stratified rows; Lyzr off)

```

Accuracy:           0.702
Balanced Accuracy:  0.729
Precision (class=1):0.380
Recall (class=1):   0.775
F1 (class=1):       0.510
Confusion matrix (rows=true, cols=pred) [0,1]:
[[547 253]
 [ 45 155]]
````



---



---

## Data

- **Source:** John Hull / Lending Club loan data  
- **Labels:** *charged off etc* → **1** (default); *fully paid* → **0** (non-default)  
- **Sample:** 1,000 stratified rows (seeded) for fast eval

---

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 eval.py      # prints metrics and writes eval.csv
````

## To run a single profile 
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 streamlit ui_pipeline.oy     
````

**Outputs**

* **stdout:** metrics + confusion matrix
* **eval.csv:** per-row results (true label, decision, scores/codes)

LLM calls are off by default for testing.

For a single profile:
Outputs decision and reasoning. 


---

## Repo

```
real_flow.py      # orchestrator (run_pipeline)
agent2.py         # features / derived metrics
agent3.py         # policy gate
agent3part2.py    # deterministic flags
agent4_infer.py   # model inference (PD, risk; SHAP top factors)
agent5.py         # decision policy (cutoff logic)
agent6.py         # explanation codes/labels
cutoff.py         # knobs (weights, cutoffs, switches)
eval.py           # 1k-row eval; prints metrics & writes eval.csv
```

```
```




    

