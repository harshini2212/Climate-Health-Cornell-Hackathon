# api


## 17:10 — decision/{severity,tau,eha,tiers,allocate}.py
allocate() + total_eha(): greedy by EHA per bucket, act-now 3 slots valued on residual risk (never >100% of a need), capacity-monotone by construction (72k-case search, 0 violations; the lazy re-valuing variant failed and its seeds are pinned). 10k veterans/day in ~105 ms. tiers.py is score-only: hazard-triggered act-now rules still TODO. make check green.

## 19:20 — api/main.py, api/store.py, outreach/export.py, allocate.compare()
All eight routes over cached parquets, POST /actions 127 ms worst case at 10,000 veterans x 120 days (budget 300); /message is 503 until outreach.messages lands; scenario and prior_scale other than sandy_then_heat / 1.0 are refused, not faked. smoke_demo.sh passes offline, make check green.

## 20:00 — allocate.headline, api/schemas.ActionRow, Week.tsx, ui fixtures
The queue card's reason line is now `headline` ("A gap in treatment, 25% chance — act today"): need, honest probability, tier urgency, and no condition, medicine or service. `rationale` and `top_driver` are untouched and still name all three for CareTeam.tsx and VeteranCard.tsx, so the drill-down lost nothing — test_the_drilldown_keeps_the_rich_rationale holds that line. xfail(strict) marker removed; the board test now also bans the string `rationale` from Week.tsx (comments stripped first) so nobody puts it back.
Partner/cohort: `headline` rides along as an allow_extra column on `actions`, **not** a contract column — CLAUDE.md gives schema.py to the cohort lane and Table.allow_extra documents exactly this, so no contract edit and nothing for you to push. It is a required field on `api.ActionRow`, so /actions fails loudly if the allocator ever stops emitting it. Also note: the ui fixtures were stale (built for 2026-06-03, a data build that no longer exists); regenerated with `--date 2026-07-16`, which is test_api's `DAY`. make_ui_fixtures defaults to `actions["date"][0]` = 2026-07-01 now, so the demo day is unpinned — not my task, but somebody should pin it.

## 01:20 — lead time: actions are scheduled by the day they must be DONE
`tau.yaml` now carries `lead_days` beside every τ row (docs/proposal.md §6's 5/2/0-day catalog): `alt_site_booking` 2, `early_refill`/`cold_chain_plan`/`pharmacist_med_review` 5, `switch_to_local_pickup`/`backup_power_plan`/`controlled_substance_bridge` 2, the two rides 1, `care_team_call`/`verified_text`/`check_in_call`/`evacuation_assist` 0 (a flash flood gives no notice, so evacuation help must still be offerable on the day). `tau.load()` is unchanged; `tau.leads()` is the new reading of the same file.

**`actions.date` is now the do-by day, and `actions.lead_days` is a contract column.** The risk day is `date + lead_days`. Capacity is spent on the do-by day, so one work day's 40 calls cover its own risk, the bookings two days out and the refills five days out at once — a live `POST /actions` for 2026-07-15 returns 347 lead-0, 15 lead-1, 155 lead-2 and 107 lead-5 rows. Do-by days share nothing, so the route is still one request per day: it reads `date .. date + 5` and returns what has to be done on `date`.

**Rahul / `ui`:** `ActionRow.lead_days: int` is required and `ActionsResponse.n_too_late: int` is new. The ribbon can now put two marks on a card — the do-by day is the response's `date`, the risk day is `date + lead_days`. `n_too_late` is the count of actions for the days ahead whose do-by day is already behind `date`; it is worth showing ("9 actions were already too late to book when the forecast arrived"). `ui/src/lib/types.ts` is yours and I did not touch it; `ui/public/fixtures/actions_candidates.json` is regenerated because `test_ui_fixtures` validates every candidate against `ActionRow`.

**Partner / `cohort`, `eval`:** I took `leeward/schema.py` this round for the one column, as prompt 13 directed — `lead_days`, Int32, bounds (0, 30). Nothing else in it moved. `compare()` now returns three values, `(actions, baseline_totals, n_too_late)`; `allocate()` is unchanged. `allocate(..., date=D)` now means "the work list for day D", not "the scores for day D", which is what `decision_quality.leeward_actions` already wants: given one day of scores it offers only the same-day actions, which is the like-for-like K-call comparison it was written for.

Two things that moved and should not surprise anyone reading the numbers:
* The marginal value chain runs for **every** veteran now, not only Act-now ones. The schedule can hand a one-slot veteran an action on Monday and another on Wednesday for the same Wednesday risk, and standalone values would then claim more harm than the veteran carries. The hand-checked five change accordingly (V4's text .00534375 not .015; total 2.77174375 not 2.7814) — their chosen set, and what it truly averts, 2.803, are unchanged.
* Scheduling costs 0.7% of harm averted on the fixture panel, against the 15% budget in the prompt. `tests/test_schedule.py` measures it against the same code with every lead set to zero, which is the only honest way to say "versus today".

Slider: 283–320 ms at 10,000 veterans, budget 300, on a machine at load 20 with three other agents' test runs on it. It was 416–480 ms before this change on the same machine — `_values()` is vectorised (the per-veteran Python loop is gone), the greedy pass reads Python lists instead of indexing numpy scalars, and `store.leads()` stops tau.yaml being re-parsed per request. `make check` green: 873 passed, 2 skipped (both the uncommitted Synthea zip).
## 00:59 — Makefile `fit`/`score`, tests/test_make_targets.py
`make score` runs the scorer that exists (rung-0 `score_prior`, then `decision.allocate` — baseline.py's own STAGES, 1.0s on fixtures) and `make fit` exits 1 naming the rung that is built and the module that would have to land, instead of `No module named`. Both resolve through `$(wildcard ...)`: when C10 lands `model/fit.py` and `model/score.py` they are picked up with no edit — verified by dropping both in and re-running.
New gate test reads every documented target's command line out of `make -n` and asserts each module it names is findable and each script it runs exists; `make demo` is checked without booting (the `module:attr` given to uvicorn must import and have that attr, `npm run dev` must exist in ui/package.json); `make setup` is excluded, being the target that creates the environment. Docs: the three now-wrong `make fit` references fixed (README table, BUILD_PLAN T+6:00); PROMPTS.md's four `make cohort && make score` lines became true rather than wrong, so they are untouched. 855 passed, 2 skipped; red only on the 300 ms /actions perf test, 320–365 ms under load avg 17–38 from five concurrent lanes, identical at HEAD before this change.

## 02:10 — the four hazard-triggered act-now rules, and prompt 10 (resolution A)

**`leeward/decision/act_now.yaml` is the new clinician-editable table**, beside `severity.yaml`
and `tau.yaml`. Four rows; each ANDs up to three cells — `veteran` (cohort), `station`
(site_status), `zip` (hazards) — and carries the `because` sentence a care team is read and a
`source`. `rules.load()` checks every column name against `leeward/schema.py` and every
operator against that column's type, so a typo fails loudly instead of quietly never firing.
On the suite's panel all four fire: mail-order-short 87 veteran-days, controlled-substance ×
SiteDown 75, unattended powered equipment 49, site-dependent × SiteDown 45.

`tiers.assign()` now returns **`tier_rule`** beside `tier`, and takes `cohort`/`hazards`/
`site_status` — all three or none; a partial set is refused, because half a rule table looks
like a full answer. The rule's sentence rides into `actions.rationale` and into the veteran
card's `why_this_tier`, replacing the probability sentence there: explaining a closed dialysis
station with a 5% number is worse than saying nothing.

**Prompt 10 is decided: A, and it needed two keys, not one.** Selection order is now
**(round, tier band, EHA)**. The band alone cost **17.3%** of total EHA — over the 15% bar —
because an Act-now veteran holds three slots and the band let them take all three before a
Find-out veteran got one, and second/third actions are valued on residual risk so they are
worth a fraction of somebody else's untouched first. Putting a **round** above the band
(everyone's first action before anyone's second) costs **5.8%** and serves the same people:
calls to Act-now veterans go 371 → 652 on the panel; on landfall day the 40 calls go 20/19/1
act_now/find_out/self_serve instead of the old self-serve-heavy split. SPEC §7.4 and §7.5,
README and PROMPTS.md all say this now.

**The band is per veteran per work day, not per candidate — that is load-bearing.** Banding
candidates breaks capacity monotonicity (a cheap Act-now action takes the slot a valuable
Self-serve one would have used, so more capacity averts less). Banding the veteran cannot,
because within a veteran the order is still descending EHA, so every action displaced by a
newly affordable one belongs to the same veteran and is worth no more than it.
`test_more_capacity_never_averts_less_harm` stays green, and there is a new per-bucket
monotonicity test over `call, ride, booking, evac, pharmacist_slot, va_fill, refill`. The
argument is informal, so I also searched it: **5,400 capacity increases over 600 seeded
Act-now-heavy instances with the rules firing, zero violations.** Recorded in the docstring
beside the search that killed the lazy-revaluing design.

**Perf: `POST /actions` costs about 5% more than before.** The first cut was **1.37×**; it is
now **1.045× at the minimum and 1.06× at p25** over 25 interleaved pairs against a
main-equivalent arm in one process. Read the fast tail, not the median: the machine has been
at load 13–60 from other lanes all session, and the median is mostly scheduler noise — it
read 1.03× at one point and 1.22× twenty minutes later on identical code, which is how I
nearly talked myself into believing a 17% regression was 3%. Four fixes got it there:
`store.act_now_rules()` caches the YAML like `tau()` does (it was being parsed twice a
request); `_rounds` drops a redundant third lexsort key; the round and the band are packed
into one integer so `lexsort` keeps three keys rather than five; and `rules.fired()` answers
each cell against its own small frame (5,340 hazard rows, 420 site rows), stays lazy so the
three joins and four rule expressions fuse into one plan, and hands the caller back its own
frame so `tiers.assign()` does not pay for a second 60k-row join. That last one alone took
`tiers.assign` from 93.7 ms to 39.7 ms at 10,000 veterans.

**Sources, corrected against the primary documents (web-verified, not assumed):**
- `docs/sources.md` claimed CMOP does "518,000 prescriptions a day" and "~129.6 million in
  FY2022". **Neither is VA's** — both trace to an uncited vendor marketing page. Replaced with
  VA's own: over 120 million a year (Dallas CMOP PIA, Oct 2024), almost 84% by mail (VA News,
  Aug 2024), 330,000 veterans a work day. The 80% share Leeward actually depends on was right.
- The controlled-substance exclusion is **confirmed twice**, including an Aug 2026 Spokane
  activation. Wording tightened: the Emergency Pharmacy Program must be *activated*, and it is
  a *participating* retail pharmacy, not any pharmacy.
- emPOWER's 36,146 / 3,165 / 1,948 re-verified live, to the unit — but they are Medicare
  beneficiaries *living independently*, and the oxygen/dialysis figures are overlapping
  subsets, not a disjoint tally. Noted in the table.
- emPOWER sets **no** multiplier for caregiver absence. That half of rule 4 now cites Casey
  et al. 2020 and Semenza et al. NEJM 1996 (living alone, OR 2.3), added to `docs/sources.md`.

**No contract file changed — nothing for anyone to pull.** `leeward/api/schemas.py` is mine
and I did not touch it: the rule reaches the UI through fields that already exist,
`VeteranCard.why_this_tier` (now the rule's own sentence when a rule fired) and
`ActionRow.rationale` (the action's sentence with the rule's appended). `leeward/schema.py`
and `model/design.py` are the cohort lane's and are untouched; `tier_rule` rides inside
`tiers.assign()`'s return and the allocator's internal frame, never into the `actions`
contract. If the UI later wants to badge *which* rule fired rather than read the sentence,
that is a `schemas.py` field and I will push it in five minutes — ask.

**Partner / `eval` — your SiteDown ablation still measures what you meant, but only just.**
`ablate.py` reaches `allocate()` through `dq.leeward_actions` with positional args and no
`hazards`/`site_status`, so no hazard rule fires there and the ablation still isolates the
*model's* SiteDown term. That is the right answer and nothing is broken. Worth knowing why it
is now a choice rather than a given: after this change SiteDown also enters the decision
outside the posterior, through `site_dependent_at_a_closed_station` and
`controlled_substance_at_a_closed_station`. If anyone ever passes the hazard frames into that
call, the ablation starts measuring the term and the rules together and will look like the
term got stronger. Keep them out, or say which one you are measuring.

**`ui/public/fixtures/*` need one more rebuild after this merges.** `1fc7b3f` rebuilt them on
the canonical opening day a few minutes before this landed, from the allocator as it was —
no band, no rounds, no hazard rules. They still validate against `ActionRow` (that model is
unchanged), so nothing goes red; the offline numbers are just from a different allocator than
the live one. This worktree has no `data/`, so I could not regenerate them honestly here.
Whoever has the real panel: re-run `make_ui_fixtures` on the same day and the offline demo
will agree with the API again.

**Rahul / `ui` — one thing to look at before the stage.** 13.7% of action rows (14.8% of
scarce-bucket rows) carry a `tier` *worse* than the band the veteran was actually worked at.
That is not the prompt-10 symptom returning: `actions.tier` is the tier of the row's **risk
day**, while the band is the veteran's most urgent tier across the risk days that work day
reaches. On landfall day the single `self_serve` call belongs to SYN-000210, who is `act_now`
about 17 July. It is a display question, not an allocation one, and it predates this change
(it arrives with do-by scheduling), but a card badged "Self-serve" holding a scarce call looks
exactly like the bug we just fixed. I did not change `actions.tier` — that column is the
cohort lane's contract and the fairness audit reads it.

**`make check` green, slider test included.** That test was red on and off all session and it
was not a clean story either way: the machine sat at load 13–60 from other lanes, and the
main-equivalent arm misses the 300 ms budget at those loads too — but the first cut of this
change *was* genuinely 1.37× slower, so for a while both things were true at once. It is
worth saying plainly: the load made the test unreliable, and the unreliable test then hid a
real regression until I stopped reading medians. It now passes at load 34. Three skips, all
`make sources-heavy` data, unchanged.
