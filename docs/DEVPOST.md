# Devpost submission copy

Ready to paste. Numbers are from `docs/BASELINES.md`, the run recorded 2026-09-20.

---

## Elevator pitch

> Care that gets ahead of the weather. Leeward turns a climate forecast into the 40 calls a VA care team can actually make today, and tells them why.

Shorter, if the field caps at 120 characters:

> A heat wave is a forecast. Leeward makes it a call list.

---

## Project story

Paste everything below into **About the project**.

### Inspiration

About 500 New Yorkers die prematurely each summer because of hot weather. Almost all of those deaths happen indoors, in homes without air conditioning running. The rise is not driven by more 95 °F days. NYC Health's 2026 heat mortality report attributes it to more **"non-extreme hot days," 82 °F up to the extreme-heat threshold**, which is why the hinge in our model sits at exactly 82 °F. The hyperparameter is a citation.

Veterans carry every heat risk factor the city counts, plus two it does not: deployment exposure and trauma.

Then we found the fact that reframed the whole project. **VA station 630, the Margaret Cochran Corbin campus in Manhattan, sits in hurricane evacuation zone 1.** It evacuated on 28 October 2012 as Sandy approached. Its opioid treatment program stayed closed for months, and around 100 veterans needed emergency guest-dosing across the city. That is not something we read in a news article; it is a join between the VHA facility registry and NYC's evacuation zones, and it runs in about two seconds. The hazard does not just reach the patient. It reaches the place the patient gets care.

So the forecast was never the missing piece. A care team with 1,200 patients and 40 calls a day already knows the storm is coming. What they do not know is **which 40**.

### What it does

Leeward answers the three questions a care team asks before a climate event, in order.

| | |
| --- | --- |
| **Who** is at risk? | A daily discrete-time hazard model, per veteran, per day, across five needs: breathing flare-up, heat illness, mental-health crisis, treatment or medication gap, and loss of access to care. |
| **When** does it peak? | Distributed lags, heat over 0–3 days and smoke over 0–2, driven by NWS and AirNow, so alerts fire days ahead rather than on the morning it is already too late. |
| **What changes if we act?** | A decision layer that scores expected harm averted for every veteran and action, then fills the team's real daily capacity like a knapsack. |

That third question is the whole project. Most risk models stop after the first and hand a care team 10,000 rows. Ten thousand rows is not a plan.

**The decision layer.** For each veteran \\(i\\), action \\(a\\), and day \\(t\\):

$$\mathrm{EHA}_{i,a,t} = \sum_{k} w_k \cdot \mathbb{E}_{\text{posterior}}\left[h_{ikt}\right] \cdot \tau_{ak}$$

where \\(w_k\\) is a clinician-set severity weight (a treatment gap for a chemotherapy patient outweighs a missed wellness text), and \\(\tau_{ak}\\) is the fraction of need \\(k\\) that action \\(a\\) prevents, from published outreach effects and adjustable in the interface. Actions are chosen greedily by harm averted per unit cost, under the team's actual budget of calls, rides, refills, bookings and pharmacist slots.

**Uncertainty is a feature, not decoration.** The posterior splits into risk we are confident about and risk we are not. A veteran with a wide interval, unknown air conditioning, unknown deployment history, gets a cheap three-minute check-in call, because the value of that information is high. A veteran with a narrow, high interval gets the expensive action: the early refill and the booked ride. Most teams will show you a confidence band. Leeward routes on it.

**The prescription says more than the diagnosis.** A diagnosis says a veteran has hypertension. The medication list says he is on hydrochlorothiazide **and** lisinopril, the exact pairing the CDC names as additive risk in heat, that he has nine days of supply left, and that those nine days arrive by mail through a ZIP that floods. In our cohort, **72.0%** are on a drug that impairs heat response and **18.4%** carry the CDC-named pair.

The sharpest edge there is a rule, not a model. **The VA emergency retail refill benefit excludes controlled substances.** A veteran can walk into any pharmacy with a VA bottle and get a ten-day supply, unless they are on an opioid, a benzodiazepine, a stimulant, or methadone through an opioid treatment program. Those are exactly the veterans the workaround does not cover, and exactly the ones Sandy stranded. Leeward surfaces them first. It never changes a medication; it flags the veteran for the VA clinical pharmacist, who decides.

**Verified outreach, because a storm is when the scammers arrive.** Every message goes only through VA channels the veteran already uses, carries a four-word verification phrase they read back to the caller, says the VA will never ask you to pay or wire money, and carries VSAFE and the Veterans Crisis Line. The interface shows that as a checklist **verified against the text being displayed, not against a flag the sender set**, so a message that lost a line in transit shows red.

It all lands on one command center: the week ahead with a countdown, the at-risk list in three tiers with risk factors and actions at a glance, a live feed of what changed, and a resource library of clinician-facing guidance for the event in progress.

### How we built it

**The model.**

$$\operatorname{logit} h_{ikt} = \alpha_k + \phi_{z(i),k} + \beta_k^{\top} H_i + \gamma_k \Lambda_i + \sum_{l=0}^{3} \delta_{kl}\, f\!\left(C_{z(i),\,t-l}\right) + \theta_k^{\top}\left(C_{z(i),t} \otimes V_i\right) + \eta_k U_i + \xi_i$$

- \\(\phi\\) is a ZIP effect with a spatial prior, so East Harlem borrows strength from Central Harlem rather than from Staten Island.
- \\(\Lambda\\) is a **latent** burn-pit exposure dose. It is never observed exactly, so it carries an era-based prior and a noisy indicator from PACT Act presumptive diagnoses. Wide uncertainty in \\(\Lambda\\) flows into wide uncertainty in risk, which is what routes that veteran to the Find-out tier.
- \\(C \otimes V\\) are the interactions that matter clinically: smoke × COPD, heat × diuretic, outage × oxygen concentrator, flood × basement, site-down × dialysis.

It fits on **binomial sufficient statistics**, not rows. Ten thousand veterans × 120 days × 5 needs is 6 million Bernoulli rows; grouping by ZIP × covariate stratum × day gives an identical likelihood with roughly fifty times fewer rows. NumPyro on JAX fits it on a laptop, the posterior is cached, and **nothing runs inference inside a request.**

We built it as a **ladder**, because a hierarchical model with spatial effects, a latent dose, distributed lags and five correlated needs is a two-day research task, not a one-day build. Rung 0 is prior-only with no MCMC and was the first thing we shipped, so from hour two onward there was always a working, defensible demo. Rung 1 is the pooled NUTS fit, rung 2 adds the interactions and the site-closure term. Every rung writes the same contract, so nothing downstream changes when you climb.

**The data: synthetic people, real places.** Every neighbourhood-level rate is real, public and cited; only the people are synthetic. We committed **22 joined reference tables** built by 18 keyless fetchers: CDC PLACES, HHS emPOWER, the NYC Heat Vulnerability Index, hurricane evacuation zones, the NYC Stormwater Flood Map, FloodNet, EPA AirNow, NWS, CDC SVI, FEMA's National Risk Index, ACS veteran counts, and the VHA facility registry. The cohort is re-homed to NYC ZIPs weighted by ACS veteran counts, and realised ZIP frequencies correlate with the source at \\(r = 0.97\\). Medications come from the RxNorm codes already on every Synthea prescription, mapped through **the VA's own 576-class drug taxonomy**, so Leeward speaks the vocabulary a VA pharmacist already uses.

Two scenarios are **replays, not scripts**. The smoke week reads real EPA monitor data for 5–11 June 2023, so the peak on screen is the actual 203.5 µg/m³ recorded at a Queens monitor, 13.1 two days before and 14.9 two days after. A judge can check the number.

**The process.** Two people, six git worktrees, one day. The architecture was the schedule: we froze the data contracts and a fixture generator in the first hour, so four lanes could develop in parallel against correctly-shaped fake data without blocking each other.

Nobody was going to read six agents' diffs, so the review had to be mechanical. `make check` is the only gate: lint, every test, and a set of **semantic guardrails** that assert the numbers *mean* the right thing. Capacity is never exceeded. Ranks are dense and ordered by harm averted. Raising capacity never lowers total harm averted. Every synthetic column carries its flag. The demo path makes no network calls. Nothing is randomised without a seed.

### Challenges we ran into

**The fit finishing by Sunday.** The real risk in a Bayesian hackathon project is not the model, it is a sampler still running at 6 a.m. We designed around it rather than hoping: binomial cells, a cached posterior, and a prior-only rung shipped first, so there was never a moment with nothing to show.

**A 4 GB surprise.** The VA Synthea release is catalogued as 10,000 synthetic veteran records, which we read as FHIR bundles. It is a 4 GB CSV archive of roughly 100,000 records. We used the 30 MB FHIR sample to prove the reader path and drew the cohort parametrically from ACS age bands, and we say that plainly rather than claiming bundles we never read.

**Three APIs that grew a key.** The Census, AirNow and VA Facilities endpoints we planned on now want credentials, which would have broken the promise that a clean clone demos offline. We found keyless equivalents for all three, committed the data, and tested with the wifi physically off.

**A label that was quietly a lie.** Our outage chip read "80% of ZIPs" when 80% was the worst single ZIP's outage fraction and 62% of ZIPs were affected. Two different numbers wearing one label, and no test would ever have caught it. We found it by reading our own screen.

**De-identification versus usefulness.** The command center hangs in a shared clinical space, so the list shows a handle like `M.H. · 0774`, the action, and counts of risk factors and suggested actions. Names, conditions and driver phrases open on a hover. A test now checks the row markup and separately checks the reveal is still gated, because a screenshot proves that once and a test proves it every time anyone edits the screen.

### Accomplishments that we're proud of

Because the cohort comes from a generator with known coefficients, we can prove the model recovers the truth we planted. That is something no team using hand-labelled synthetic profiles can do.

| Check | Result |
| --- | --- |
| **Harm averted at 40 calls/day** | **16.14** vs 6.79 for the best baseline (rank by chronic-condition count), 3.37 for rank-by-age, 3.74 for random. **2.38× the best baseline.** |
| **Per call actually made** | 0.5185, a **3.05×** ratio, spending 934 of 1,200 available calls |
| **Parameter recovery** | **98.3%** of true coefficients inside the 90% posterior interval, against a 90% bar |
| **Calibration (ECE)** | max 0.0076 against a 0.03 bar |
| **Discrimination (within-day AUC)** | heat 0.727, treatment gap 0.715 |
| **Fairness** | 0 groups flagged of 33, worst false-negative ratio 1.048 |

The thing we are most proud of is not a number, it is two refusals to flatter ourselves.

**We said calibration alone proves nothing here.** At these base rates, a single constant equal to the base rate scores a perfect ECE of 0.0000. We drew that reference line on the chart instead of quietly banking the number, and added within-day AUC, because the only ranking question a care team acts on is "of today's panel, who?".

**We made the fairness audit a screen, not a footnote,** and a failing audit is displayed, never suppressed. Black New Yorkers die of heat stress at roughly three times the rate of white New Yorkers, so an unaudited model here is not a neutral object.

And the numbers above come from the **prior-only rung**. That is the floor, not the ceiling.

### What we learned

**Say what a number means on the screen it is on.** We shipped two different "harm averted" figures, one expected and one realised, that disagreed by a factor of four. Both were correct. We had to label them in place, because "which one is true?" is a question you cannot answer well from a stage.

**The demo has to open where the story is.** Day 0 of our scenario is nine weeks before anything happens, so the first thing anyone saw was an empty ribbon under "No active alerts in the 7-day window." A true answer to the wrong question. The opening window is now computed from the hazard table itself, two days in front of the first alert, so it stays right when the dates move.

**A skipping test is a to-do list.** Our best trick: a guardrail for a module that does not exist yet skips with a message naming what it will enforce, and starts enforcing the instant that module lands. The suite tightened itself as the build filled in, and nobody had to remember to turn anything on.

**Build the floor before the ceiling.** Shipping the prior-only rung in the first two hours meant every later decision was an upgrade rather than a gamble.

### What's next for Grounded Cyborgs

1. **Validate the priors on real data.** Work with a VA research partner under IRB on de-identified records, replacing planted truth with real outcomes.
2. **Pilot one summer with one NYC VA medical center care team.** The success metric is harm averted at fixed staff time, measured against the prior summer, not model accuracy.
3. **Close the loop.** The outcome log already records whether each call happened, whether the veteran was reached, and whether the need occurred. Twenty to thirty rows a week is exactly what a Bayesian model absorbs without a retrain, which is how the literature-derived priors get tested rather than trusted.
4. **Open-source the cohort generator and the evaluation harness,** so other teams can benchmark against a shared synthetic NYC cohort instead of each inventing their own.

**What we will not claim.** Nothing here validates the model on real veterans. It validates that the machinery is correct and the uncertainty is honest, which is the precondition for a pilot, not a substitute for one. Intervention effects are literature priors, adjustable, not measured here. The tool proposes; clinicians approve every high-tier action.

---

## Built with

```
python, numpyro, jax, bayesian-inference, polars, fastapi, pydantic, uvicorn,
react, typescript, vite, deck.gl, arviz, plotly, numpy, pandas, pyarrow,
geopandas, shapely, pytest, ruff, github-actions, synthea, fhir, rxnorm
```

If the form prefers fewer, the eight that carry the story:

```
python, numpyro, jax, fastapi, react, typescript, deck.gl, polars
```
