
# BNPL Credit Risk — agentic pipeline

**Goal:** build an agentic BNPL risk pipeline and optimize **Balanced Accuracy** (BA) end-to-end on a 1k stratified sample. **Target BA > 0.75.**

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

final\_score = 0.8 \* ML\_risk + 0.2 \* flag\_points

```
- **Flags:** max 5 per profile  
  - Red: **+12** (high), **+10** (med), **+8** (low)  
  - Green: **−15** (high), **−10** (med), **−5** (low)
- In this eval, **LLM flags are off**.
- **Cutoff & rule:** `final_score ≥ 23 → REJECT`, else APPROVE.
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

Accuracy:           0.413
Balanced Accuracy:  0.609
Precision (class=1):0.246
Recall (class=1):   0.935
F1 (class=1):       0.389
Confusion matrix \[rows=true (0,1); cols=pred (0,1)]:
\[\[226 574]
\[ 13 187]]

````

**Errors:** 738 / 1000 rows

---

## What I tried (didn’t move BA)

- **Import/cache cleanup:** earlier **0.725 BA** was a mirage from mixed imports.  
- **Changing cutoffs:** adjusted across 20–60; didn’t improve BA (best single run at cutoff **23** hit ~**0.602 BA**).  
- **Gate & flags variants:** tightened/loosened gate.  
- **Alt model:** swapped classifier → **lower BA** than current HGBM; reverted.  
- **Weights:** different ML vs flags splits (e.g., **0.7/0.3** and **0.9/0.1** ML weight) — no durable BA lift.

---

## Data

- **Source:** John Hull / Lending Club loan data  
- **Labels:** *charged off etc* → **1** (default); *fully paid* → **0** (non-default)  
- **Sample:** 1,000 stratified rows (seeded) for fast eval

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eval.py      # prints metrics and writes eval.csv
````

**Outputs**

* **stdout:** metrics + confusion matrix
* **eval.csv:** per-row results (true label, decision, scores/codes)

LLM calls are off by default for testing.

---

## Known issues

* **Old BA 0.72 ≠ real:** stale imports/global state; cleared.
* multiple errors during eval (refer `eval_results.csv`)
* **Model details:** see `agent4.csv` for configuration/notes (HGBM + SHAP).

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




    

