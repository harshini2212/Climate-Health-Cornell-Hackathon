# Devpost submission copy

Ready to paste. Numbers are from `docs/BASELINES.md`, the run recorded 2026-09-20.

---

## Elevator pitch

> Care that gets ahead of the weather. Leeward turns a climate forecast into the 40 calls a VA care team can actually make today, and tells them why.

Alternates, if the first runs long:

- Leeward turns a climate forecast into a ranked call list for VA care teams: who to reach, when, and why, cut at the team's real capacity.
- A heat wave is a forecast. Leeward makes it a call list.

---

## About the project

### The problem is not that nobody saw it coming

About 500 New Yorkers die prematurely each summer because of hot weather. Almost all of those deaths happen indoors, in homes without air conditioning running. The rise is not driven by more 95 °F days. NYC Health's 2026 heat mortality report attributes it to more **"non-extreme hot days," 82 °F up to the extreme-heat threshold**, which is why the hinge in our model sits at exactly 82 °F. The hyperparameter is a citation.

Veterans carry every heat risk factor the city counts, plus two it does not: deployment exposure and trauma.

And the hazard reaches the care site itself. **VA station 630, the Margaret Cochran Corbin campus in Manhattan, sits in hurricane evacuation zone 1.** It evacuated on 28 October 2012 as Sandy approached. Its opioid treatment program stayed closed for months, and around 100 veterans needed emergency guest-dosing across the city. That is not a story we found in a news article; it is a join between the VHA facility registry and NYC's evacuation zones, and it runs in about two seconds.

The forecast was never the missing piece. A care team with 1,200 patients and 40 calls a day already knows the storm is coming. What they do not know is **which 40**.

### What Leeward does

Leeward answers the three questions a care team asks before a climate event, in order.

| | |
| --- | --- |
| **Who** is at risk? | A daily discrete-time hazard model, per veteran, per day, across five needs: breathing flare-up, heat illness, mental-health crisis, treatment or medication gap, and loss of access to care. |
| **When** does it peak? | Distributed lags, heat over 0–3 days and smoke over 0–2, driven by NWS and AirNow, so alerts fire days ahead rather than on the morning it is already too late. |
| **What changes if we act?** | A decision layer that scores expected harm averted for every veteran and action, then fills the team's real daily capacity like a knapsack. |

That third question is the whole project. Most risk models stop after the first and hand a care team 10,000 rows. Ten thousand rows is not a plan.

### The decision layer

For each veteran \\(i\\), action \\(a\\), and day \\(t\\):

$$\mathrm{EHA}_{i,a,t} = \sum_{k} w_k \cdot \mathbb{E}_{\text{posterior}}\left[h_{ikt}\right] \cdot \tau_{ak}$$

where \\(w_k\\) is a clinician-set severity weight (a treatment gap for a chemotherapy patient outweighs a missed wellness text), and \\(\tau_{ak}\\) is the fraction of need \\(k\\) that action \\(a\\) prevents, taken from published outreach effects and adjustable in the interface. Actions are then chosen greedily by harm averted per unit cost, under the team's actual budget of calls, rides, refills, bookings and pharmacist slots.

**Uncertainty is a feature, not decoration.** The posterior splits into risk we are confident about and risk we are not. A veteran with a wide interval, unknown air conditioning, unknown deployment history, gets a cheap three-minute check-in call, because the value of that information is high. A veteran with a narrow, high interval gets the expensive action: the early refill and the booked ride. Most teams will show you a confidence band. Leeward routes on it.

### The model

$$\operatorname{logit} h_{ikt} = \alpha_k + \phi_{z(i),k} + \beta_k^{\top} H_i + \gamma_k \Lambda_i + \sum_{l=0}^{3} \delta_{kl}\, f\!\left(C_{z(i),\,t-l}\right) + \theta_k^{\top}\left(C_{z(i),t} \otimes V_i\right) + \eta_k U_i + \xi_i$$

- \\(\phi\\) is a ZIP effect with a spatial prior, so East Harlem borrows strength from Central Harlem rather than from Staten Island.
- \\(\Lambda\\) is a **latent** burn-pit exposure dose. It is never observed exactly, so it carries an era-based prior and a noisy indicator from PACT Act presumptive diagnoses. Wide uncertainty in \\(\Lambda\\) flows into wide uncertainty in risk, which is what routes that veteran to the Find-out tier.
- \\(C \otimes V\\) are the interactions that matter clinically: smoke × COPD, heat × diuretic, outage × oxygen concentrator, flood × basement, site-down × dialysis.

It is fitted on **binomial sufficient statistics**, not rows. Ten thousand veterans × 120 days × 5 needs is 6 million Bernoulli rows; grouping by ZIP × covariate stratum × day gives an identical likelihood with roughly fifty times fewer rows. NumPyro on JAX fits it on a laptop, the posterior is cached, and **nothing runs inference inside a request.**

We built the model as a **ladder**, because a hierarchical model with spatial effects, a latent dose, distributed lags and five correlated needs is a two-day research task, not a one-day build. Rung 0 is prior-only with no MCMC and was the first thing we shipped, so from hour two onward there was always a working, defensible demo. Rung 1 is the pooled NUTS fit, rung 2 adds the interactions and the site-closure term. Every rung writes the same contract, so nothing downstream changes when you climb.

### The prescription says more than the diagnosis

A diagnosis says a veteran has hypertension. The medication list says he is on hydrochlorothiazide **and** lisinopril, the exact pairing the CDC names as additive risk in heat, that he has nine days of supply left, and that those nine days arrive by mail through a ZIP that floods.

Synthea puts an RxNorm code on every prescription, and RxNav publishes the **VA's own 576-class drug taxonomy**, keyless. So Leeward speaks the vocabulary a VA clinical pharmacist already uses. In our cohort:

| | |
| --- | --- |
| On a medication that impairs heat response | **72.0%** |
| On the ACE-inhibitor-or-ARB plus diuretic pair the CDC names | **18.4%** |
| Cold chain (insulin and friends) | 10.8% |
| Controlled substance | 5.5% |

That last row is the sharp edge, and it is a rule rather than a model. **The VA emergency retail refill benefit excludes controlled substances.** A veteran can walk into any pharmacy with a VA bottle and get a ten-day supply, unless they are on an opioid, a benzodiazepine, a stimulant, or methadone through an opioid treatment program. Those are exactly the veterans the workaround does not cover, and exactly the ones Sandy stranded. Leeward surfaces them first.

**Leeward never changes a medication.** It flags a veteran for the VA clinical pharmacist, who decides. `pharmacist_slot` is a scarce capacity unit in the allocator for precisely that reason: it is a real person's afternoon.

### Synthetic people, real places

Every neighbourhood-level rate in this project is real, public and cited. Only the people are synthetic.

We committed **22 joined reference tables** built by 18 keyless fetchers: CDC PLACES, HHS emPOWER, the NYC Heat Vulnerability Index, hurricane evacuation zones, the NYC Stormwater Flood Map, FloodNet, EPA AirNow, NWS, CDC SVI, FEMA's National Risk Index, ACS veteran counts, and the VHA facility registry. The cohort is re-homed to NYC ZIPs weighted by ACS veteran counts, and the realised ZIP frequencies correlate with the source at \\(r = 0.97\\).

Two scenarios are **replays, not scripts**. The smoke week reads real EPA monitor data for 5–11 June 2023, so the peak on screen is the actual 203.5 µg/m³ recorded at a Queens monitor, 13.1 two days before and 14.9 two days after. A judge can check the number.

What is genuinely synthetic, housing floor, burn-pit years, per-person air conditioning, days of supply, and every daily outcome, carries a `_synthetic` flag enforced by a test.

### How we know it works

The cohort comes from a generator with known coefficients, so we can prove the model recovers the truth we planted. That is something no team using hand-labelled synthetic profiles can do.

| Check | Result |
| --- | --- |
| **Harm averted at 40 calls/day** | **16.14** vs 6.79 for the best baseline (rank by chronic-condition count), 3.37 for rank-by-age, 3.74 for random. **2.38× the best baseline.** |
| **Per call actually made** | 0.5185, a **3.05×** ratio, spending 934 of 1,200 available calls |
| **Parameter recovery** | **98.3%** of true coefficients inside the 90% posterior interval, against a 90% bar |
| **Calibration (ECE)** | max 0.0076 against a 0.03 bar |
| **Discrimination (within-day AUC)** | heat 0.727, treatment gap 0.715 |
| **Fairness** | 0 groups flagged of 33, worst false-negative ratio 1.048 |

Two things we insisted on saying out loud.

**Calibration alone proves nothing here.** At these base rates, a single constant equal to the base rate scores an ECE of 0.0000. We put that reference line on the chart rather than quietly banking the number, and added within-day AUC because the only ranking question a care team acts on is "of today's panel, who?".

**The fairness audit is a screen, not a footnote,** and a failing audit is displayed, never suppressed. Black New Yorkers die of heat stress at roughly three times the rate of white New Yorkers, so an unaudited model here is not a neutral object.

The headline numbers above come from the **prior-only rung**. That is the floor, not the ceiling.

### Verified outreach, because a storm is when the scammers arrive

Scammers pose as FEMA staff, adjusters and charities after disasters. Every Leeward message is built so a veteran can tell it apart:

- It goes only through VA channels the veteran already uses: VEText, My HealtheVet secure message, the care-team phone. Never a new number.
- It carries a **four-word verification phrase** the veteran reads back to the caller.
- It says the VA will never ask you to pay, wire money, or share bank details.
- It carries VSAFE, the VA fraud line, and the Veterans Crisis Line.

The interface renders that as a visible checklist, **verified against the text being displayed, not against a flag the sender set**, so a message that lost a line in transit shows red. Tests enforce all five elements, block URL shorteners, and reject any phone number outside the allow-list.

### How we built it

Two people, six git worktrees, one day. The architecture was the schedule: we froze the data contracts and a fixture generator in the first hour, which let four lanes develop in parallel against correctly-shaped fake data without blocking each other.

Nobody was going to read six agents' diffs, so the review had to be mechanical. `make check` is the only gate: lint, every test, and a set of **semantic guardrails** that assert the numbers *mean* the right thing. Capacity is never exceeded. Ranks are dense and ordered by harm averted. Raising capacity never lowers total harm averted. Every synthetic column carries its flag. The demo path makes no network calls. Nothing is randomised without a seed.

The best trick: a guardrail for a module that does not exist yet **skips with a message naming what it will enforce, and starts enforcing the instant that module lands.** The skip list is a live, accurate to-do list that tightens itself as the build fills in.

### What we learned

**Say what a number means, on the screen it is on.** We shipped two different "harm averted" figures, one expected and one realised, and they disagreed by a factor of four. Both were correct. We had to label them in place, because "which one is true?" is a question you cannot answer well from the stage.

**A label is a claim.** Our outage chip read "80% of ZIPs" when 80% was the worst single ZIP's outage fraction and 62% of ZIPs were affected. Two different numbers wearing one label. Found by reading our own screen, not by a test.

**The demo opens where the story is.** Day 0 of the scenario is nine weeks before anything happens, so the first thing anyone saw was an empty ribbon under "No active alerts." We now compute the opening window from the hazard table itself, two days in front of the first alert, which stays correct when the dates move.

**Offline by construction, not by hope.** No API key, anywhere. Three upstream APIs we planned on now want credentials; we found keyless equivalents for all three, committed the data, and tested with the wifi physically off.

### Challenges

**The fit finishing by Sunday.** Mitigated by design rather than by luck: binomial cells, a cached posterior, and a prior-only rung shipped first so there was never a moment with nothing to show.

**A 4 GB surprise.** The VA Synthea release is not 10,000 FHIR bundles, as the catalogue implies, but a 4 GB CSV archive of roughly 100,000 records. We used the 30 MB FHIR sample to prove the reader path and drew the cohort parametrically from ACS age bands, and we say so rather than claiming bundles we did not read.

**De-identification versus usefulness.** The command center hangs in a shared clinical space, so the list shows a handle like `M.H. · 0774`, the action, and counts of risk factors and suggested actions. Names, conditions and driver phrases open on a hover. A test checks the row markup and separately checks the reveal is still gated, because a screenshot proves this once and a test proves it every time anyone edits the screen.

### What we will not claim

Nothing here validates the model on real veterans. It validates that the machinery is correct and the uncertainty is honest, which is the precondition for a pilot, not a substitute for one. Priors from WTC Registry and PACT Act evidence set centres only, and the ranking is robust to halving or doubling them. Intervention effects are literature priors, adjustable, not measured here. The tool proposes; clinicians approve every high-tier action.

### What is next

1. Validate the priors on de-identified VA data with a research partner under IRB, replacing planted truth with real outcomes.
2. Pilot one summer with one NYC VA medical center care team. Success is harm averted at fixed staff time, measured against the prior summer.
3. Publish the cohort generator and evaluation harness so other teams can benchmark against a shared synthetic NYC cohort.

---

## Built with

Up to 25 tags. Paste as a comma-separated list.

```
python, numpyro, jax, bayesian-inference, polars, fastapi, pydantic, uvicorn,
react, typescript, vite, deck.gl, arviz, plotly, numpy, pandas, pyarrow,
geopandas, shapely, pytest, ruff, github-actions, synthea, fhir, rxnorm
```

If the form prefers fewer, the eight that carry the story:

```
python, numpyro, jax, fastapi, react, typescript, deck.gl, polars
```
