# Running and demoing Leeward

Everything here works with the wifi off. If a step needs the network, it says so.

---

## 1. Setup — once, about two minutes

```bash
git clone git@github.com:harshini2212/Climate-Health-Cornell-Hackathon.git
cd Climate-Health-Cornell-Hackathon
make setup          # venv + deps + editable install
make check          # THE GATE: lint + 1,019 tests. Must be green.
```

`data/reference/` (21 public tables) and `ui/dist` (the built bundle) are committed, so a
clean clone can boot without fetching or building anything.

---

## 2. Build the data — about 40 seconds

```bash
make hazards        # scenario + data/reference -> hazards.parquet, site_status.parquet
make cohort         # 10,000 veterans, then 120 days of simulated outcomes
make score          # scores.parquet + actions.parquet
make report         # calibration, recovery, fairness, discrimination -> report/report.json
```

Or all of it, plus a recorded row in `docs/BASELINES.md`:

```bash
make baseline LABEL="my run"
make baseline-diff          # what moved since the last one
```

Everything is seeded. The same command produces the same numbers — verified by running the
whole pipeline twice and diffing every tracked metric.

---

## 3. Run it

**The way to demo it.** One process, the built bundle, no node, no network:

```bash
make demo DAY=63
```

→ **http://127.0.0.1:8000/?day=63**

**The way to work on the UI.** Vite dev server with hot reload, API behind a proxy:

```bash
make demo-dev DAY=63     # needs `cd ui && npm install` once
```

→ **http://localhost:5173/?day=63**

After editing anything in `ui/src`, rebuild and commit the bundle or `make demo` serves stale
code: `make ui`.

### Why `DAY=63`

The scenario runs 2026-06-01 → 2026-09-28. Day 0 is a calm week and the board is nearly
empty — true to life, useless as an opening.

| DAY | What is happening |
| --- | --- |
| `0` | Calm week. Useful only as a deliberate contrast. |
| **`63`** | **Landfall, 3 Aug. Coastal flood warning, station 630 closes.** Open here. |
| `67` | The heat wave — 101 °F on 7 Aug, the day with the most work on it |

Station **630** — the Manhattan VA — is down from **3 Aug to 16 Sep**, 45 days.

---

## 4. The demo, screen by screen

Five screens, about ninety seconds. Say the word **"synthetic"** early and at least twice.

### Week board — the default screen

Seven day columns with hazard glyphs, a closure banner, the queue under the day the work is
**due**, owner lanes, and the count of who does not fit at this capacity.

> "This is what a VA care team sees on a screen beside them. Not a risk score — next week's
> work, in the order it has to happen. Thursday is heavy because of that." *(point at the
> surge glyph)*

Note the handles: **`A.J. · 6219`**, not names. It is designed to hang in a shared clinical
space, so conditions and medicines stay behind a click.

### Click 3 August — the care-team list

Cut at forty calls. Rank, tier, the one-line reason, owner, EHA.

> "Forty calls is what this team can actually make. These are the forty."

### Move the capacity slider — 40 → 20 → 80

The best three seconds you have. The list re-ranks and the harm-averted counter moves.

> "At twenty, these twenty. At eighty, these. The question was never who is at risk — it is
> who we can reach, and what changes if we do."

### Open a veteran card — `SYN-006219`

**Andre Jackson, 66, Bronx.** On station 630's opioid treatment program. Eight active
medications. Action: **evacuation assist**, tier act-now.

Five needs as 10–90% interval bars with the epistemic share shaded, drivers as plain-language
chips, the medication panel, one "why this tier" line.

> "When the Manhattan VA closed for Sandy in 2012, its opioid treatment program stayed shut
> five months and about a hundred veterans needed emergency dosing across the city. We have
> a hundred and seven on methadone; eighty-one of them are at station 630."

### Open his message

> "Sent through a channel he already uses. Four words he can read back — *diamond tempo
> violet wax*. The VA will never ask you to pay. VSAFE. 988 press 1. A scammer cannot
> reproduce that."

### Model report — if a judge asks how you know

Recovery, reliability curves, harm averted against three baselines, ablations, fairness.

---

## 5. Testing it yourself

```bash
make check          # lint + 1,019 tests
make smoke          # boots the API with the network blocked, hits every route
make clean-clone    # proves a fresh clone boots and serves offline in under 60s
make status         # module map, table row counts, model rung, gate state
```

Every route by hand, with the API up on 8000:

```bash
curl "127.0.0.1:8000/forecast?scenario=sandy_then_heat&day=63" | jq '.headline, (.facilities|map(select(.site_down))|length)'
curl "127.0.0.1:8000/scores?date=2026-08-03&need=heat"          | jq '.zips|length'
curl "127.0.0.1:8000/veteran/SYN-006219?date=2026-08-03"        | jq '.name_display, .tier, .medications'
curl -X POST 127.0.0.1:8000/actions -H 'content-type: application/json' \
     -d '{"date":"2026-08-03","capacity":{"call":40,"refill":200,"ride":15,"booking":20,"evac":8,"partner_slot":10,"pharmacist_slot":12,"va_fill":30,"free":10000}}' \
     | jq '.total_eha, .counts_by_tier, (.actions|length)'
curl "127.0.0.1:8000/message/msg-ea6c9816821e"                  | jq '.verification_phrase, .includes_vsafe'
curl 127.0.0.1:8000/report                                      | jq '.ece_by_need, .constant_ece, .fairness_failed'
curl "127.0.0.1:8000/export?date=2026-08-03" | head -3
```

**The one to run before you believe any calibration claim:**

```bash
curl -s 127.0.0.1:8000/report | jq '{model: .ece_by_need, control: .constant_ece}'
```

A constant at the base rate scores **0.0000 on every need** — better than the model. ECE was
never evidence on its own. The evidence is `discrimination`.

---

## 6. The numbers, and which ones survive scrutiny

**Quote these.**

| | |
| --- | --- |
| Harm averted at 40 calls/day | **489.4 on 934 calls** vs 203.7 on all 1,200 — 2.4× the harm on **22% fewer calls**, **3.05× per call** |
| Discrimination, heat | within-day AUC **0.727**, lift@1% **10.5×** |
| Discrimination, treatment gap | AUC **0.715**, lift@1% **15.3×** |
| Find-out tier | **9.15%** of actions — up from 0.14% before honest missingness |
| Medication | 4.4 drugs each · **72%** on a heat-impairing drug · **18.4%** on the pair CDC names · 547 flagged for the pharmacist |
| Panel | 10,000 veterans, 175 real NYC ZIPs, ZIP weights correlating **0.97** with ACS |

**Do not quote these without the caveat.**

- **ECE 0.0076.** A constant at the base rate beats it. Necessary, not sufficient.
- **"0 of 33 fairness groups flagged."** At this cohort's pooled FNR the bar is unreachable,
  so it has no failing state. What the audit *does* show: Staten Island is reached at half
  the cohort rate with 13% of the coverage.
- **Five needs.** Only heat and treatment gap discriminate. Access-loss and mental have
  **negative scaled Brier** — worse than predicting the base rate. Say two of five.
- **"4.2× better."** That was measured when the allocator spent all forty calls every day
  including calm ones. The honest number is 2.4× on 22% fewer calls.

**If asked what model fitted:** rung 0, prior-only. Rung 1 (pooled NUTS, r-hat max 1.0082,
0 divergences, ESS min 1,390) is in PR #25 and does not change what the demo serves.

---

## 7. When it goes wrong

| Symptom | Cause | Fix |
| --- | --- | --- |
| Board is empty | Opened on a calm week | `?day=63` |
| UI looks stale after editing `ui/src` | `ui/dist` is the committed bundle | `make ui` |
| `make check` red right after `make cohort` | You are on an old checkout; tests own their data now | `git pull`, `make check` |
| Slider feels slow | The API and a Vite dev server are competing | Use `make demo`, not `make demo-dev` |
| A route 500s | `data/*.parquet` half-written | Re-run §2 top to bottom |
| Port 8000 busy | An earlier API is still up | `pkill -f "uvicorn leeward.api.main"` |

**Before you present:** `make demo DAY=63`, click every screen once, leave the tab open. Cold
JIT on the first deck.gl render is a five-second silence at the worst moment.
