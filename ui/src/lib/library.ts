/**
 * The resource library: what a care team needs to know about climate events and the
 * patients most exposed to them. Every entry names its source; the numbers here are the
 * public ones the repo already cites in docs/sources.md. Nothing here is medical advice
 * to a patient: it is the clinician-facing guidance the team acts on.
 */

export type EventKind = "heat" | "smoke" | "flood" | "outage" | "all";
export type Topic = "medications" | "conditions" | "programs" | "outreach" | "housing";

export interface Resource {
  id: string;
  event: EventKind;
  topic: Topic;
  title: string;
  summary: string;
  points: string[];
  source: string;
  /** Words the Ask panel matches on, beyond the title. */
  keywords: string[];
}

export const EVENT_LABEL: Record<EventKind, string> = {
  heat: "Extreme heat",
  smoke: "Wildfire smoke",
  flood: "Coastal and flash flooding",
  outage: "Power outage",
  all: "Every event",
};

export const TOPIC_LABEL: Record<Topic, string> = {
  medications: "Medications",
  conditions: "Conditions",
  programs: "VA programs",
  outreach: "Outreach and scams",
  housing: "Housing and evacuation",
};

export const LIBRARY: Resource[] = [
  {
    id: "heat-meds",
    event: "heat",
    topic: "medications",
    title: "Medications that impair heat response",
    summary: "Diuretics, anticholinergics, beta-blockers and antipsychotics blunt sweating, thirst or cardiac output. The CDC names the ACE-inhibitor-or-ARB plus diuretic pair as an additive risk.",
    points: [
      "Review the medication list before, not during, a heat alert; CDC guidance is to plan dose timing and hydration with the prescriber.",
      "Diuretics: dehydration and electrolyte loss; hydrochlorothiazide with lisinopril is the most common pairing in the cohort.",
      "Anticholinergics (ACB score ≥ 3): reduced sweating and heat dissipation.",
      "Beta-blockers: blunted cardiac response to heat stress.",
      "Antipsychotics and some antidepressants: impaired thermoregulation and thirst.",
      "Leeward never changes a dose. It flags the veteran for the VA clinical pharmacist, who decides.",
    ],
    source: "CDC, Heat and Medications: Guidance for Clinicians",
    keywords: ["diuretic", "hydrochlorothiazide", "lisinopril", "anticholinergic", "beta-blocker", "antipsychotic", "acb", "pharmacist", "meds"],
  },
  {
    id: "heat-82",
    event: "heat",
    topic: "conditions",
    title: "Why the hinge is 82 °F, not 95 °F",
    summary: "NYC Health attributes the rise in heat-exacerbated deaths mainly to more non-extreme hot days, 82 °F up to the extreme-heat threshold. About 500 New Yorkers die prematurely each summer from heat; most deaths happen at home without air conditioning running.",
    points: [
      "Roughly 490 heat-exacerbated deaths a year (2014–2023) and about 7 heat-stress deaths a year; 19 in the June 2025 heat wave alone.",
      "Indoor deaths are almost always in homes without AC running; owning an AC and being unable to afford to run it is the same risk.",
      "Black New Yorkers die of heat stress at three times the rate of white New Yorkers, which is why the fairness audit is a screen, not a footnote.",
      "Heat harm lags exposure by 0–3 days: the call has to happen before the peak, not on it.",
    ],
    source: "NYC Health, 2026 Heat-Related Mortality Report",
    keywords: ["82", "hot day", "mortality", "air conditioning", "ac", "heat index", "deaths"],
  },
  {
    id: "heat-chf-ckd",
    event: "heat",
    topic: "conditions",
    title: "Heart failure, kidney disease and heat",
    summary: "Fluid balance is the whole problem: heart-failure patients on fluid restriction and dialysis patients between sessions have the least margin when it is hot.",
    points: [
      "Heart failure: dehydration versus fluid restriction is a prescriber conversation, not a text message.",
      "Dialysis: interdialytic weight gain rises with thirst; missed sessions during an event compound quickly.",
      "Diabetes: insulin absorption changes with heat, and insulin is cold-chain.",
      "Older adults (65+) and people living alone are the two strongest non-clinical predictors in the city's data.",
    ],
    source: "CDC heat guidance; NYC Health heat mortality report",
    keywords: ["chf", "heart failure", "ckd", "kidney", "dialysis", "diabetes", "insulin", "older", "65"],
  },
  {
    id: "smoke-copd",
    event: "smoke",
    topic: "conditions",
    title: "Wildfire smoke, COPD and asthma",
    summary: "PM2.5 from wildfire smoke drives breathing flare-ups within 0–2 days. In June 2023 the citywide peak was 203.5 µg/m³ (AQI 254) at a Queens monitor; it was 13 two days earlier and 15 two days later.",
    points: [
      "AQI above 150 is unhealthy for everyone; above 100 the sensitive groups (COPD, asthma, heart disease, children, older adults) should limit outdoor exertion.",
      "Rescue inhaler on hand and a supply check are the first calls; a clean-air room with a HEPA unit is the action.",
      "Keep windows closed; a well-fitted N95 for unavoidable time outdoors.",
      "PACT Act presumptive respiratory conditions from burn-pit exposure sit in this same group.",
    ],
    source: "EPA AirNow; CDC wildfire smoke guidance; VA PACT Act",
    keywords: ["copd", "asthma", "pm2.5", "aqi", "inhaler", "breathing", "pact", "burn pit", "hepa", "n95"],
  },
  {
    id: "flood-site",
    event: "flood",
    topic: "programs",
    title: "When a VA site closes: dialysis, infusion, OTP",
    summary: "Station 630, the Manhattan VA campus, sits in hurricane evacuation zone 1. It evacuated ahead of Sandy on 28 October 2012; its opioid treatment program stayed closed for months and about 100 veterans needed emergency guest-dosing across the city.",
    points: [
      "Site-dependent care cannot be texted: dialysis, chemotherapy infusion and methadone dosing need an alternate site booked before the closure.",
      "Brooklyn (630A4) and the Bronx (526) carry dialysis; Manhattan carries all three services.",
      "Book the alternate site and the transport together; a booking without a ride is not a plan.",
      "Guest-dosing for OTP patients requires coordination between programs; start it at the flood watch, not the warning.",
    ],
    source: "VA facility registry (VHA ArcGIS); NYC hurricane evacuation zones; VA Sandy after-action reporting",
    keywords: ["630", "manhattan", "closed", "site", "dialysis", "infusion", "otp", "methadone", "evacuation zone", "sandy"],
  },
  {
    id: "flood-mail",
    event: "flood",
    topic: "medications",
    title: "Four in five VA prescriptions arrive by mail",
    summary: "VA's Consolidated Mail Outpatient Pharmacy delivers about 80 percent of outpatient prescriptions, roughly 518,000 a day. A flooded or evacuated ZIP is a medication-supply event, not only a clinic event.",
    points: [
      "Anyone under ten days of supply in a disrupted ZIP gets an early refill or a switch to local pickup.",
      "The VA emergency retail refill benefit lets a veteran get a 10-day supply at a retail pharmacy with a VA bottle.",
      "That benefit excludes controlled substances: opioids, benzodiazepines, stimulants and OTP methadone need a VA fill or a bridge, which is why they rank first.",
      "Cold-chain drugs (insulin) need a plan for the outage, not just the flood.",
    ],
    source: "VA Pharmacy Benefits Management; VA CMOP; VA emergency prescription guidance",
    keywords: ["mail", "cmop", "refill", "supply", "controlled", "opioid", "benzodiazepine", "pharmacy", "retail"],
  },
  {
    id: "outage-equipment",
    event: "outage",
    topic: "conditions",
    title: "Electricity-dependent patients in an outage",
    summary: "HHS emPOWER counts 36,146 electricity-dependent Medicare beneficiaries in NYC, including 3,165 on oxygen and 1,948 on facility dialysis. Oxygen concentrators, ventilators, home dialysis and powered beds fail with the grid.",
    points: [
      "A backup-power plan before the event: cylinders for oxygen, battery hours known, the utility's medical-needs registry.",
      "A care-team call within two hours of an outage for anyone on powered equipment.",
      "Evacuation to a site with power beats waiting out a multi-day outage; the Sandy outage lasted four days along the waterfront.",
      "Insulin and other refrigerated medicines: a cold-chain plan with the pharmacist.",
    ],
    source: "HHS emPOWER; CDC power outage guidance",
    keywords: ["outage", "power", "oxygen", "concentrator", "ventilator", "empower", "battery", "cold chain", "insulin"],
  },
  {
    id: "flood-housing",
    event: "flood",
    topic: "housing",
    title: "Basements, ground floors and evacuation zones",
    summary: "Eleven of the thirteen New Yorkers who died in Ida's flash flood on 1 September 2021 were in basement apartments. Zone 1 is ordered to evacuate first; zones run 1 to 6.",
    points: [
      "Basement and ground-floor homes in a flooding ZIP, with mobility limits, go first for evacuation assistance.",
      "Evacuation centres with step-free access; the pack list is medicines, equipment, chargers and IDs.",
      "FloodNet street sensors and the NYC Stormwater Flood Map say which ZIPs flood in a moderate-rain event; the Rockaways, Coney Island and Lower Manhattan lead.",
      "A caregiver at the same address can act the same day; no caregiver on record moves the veteran up the list.",
    ],
    source: "NYC Emergency Management hurricane evacuation zones; NYC Stormwater Flood Map; FloodNet",
    keywords: ["basement", "ground floor", "evacuate", "evacuation", "zone", "ida", "floodnet", "caregiver", "mobility"],
  },
  {
    id: "mental-988",
    event: "all",
    topic: "conditions",
    title: "Mental health during an event",
    summary: "Trauma exposure raises the odds that a storm or a blackout becomes a crisis. Any crisis signal goes to the Veterans Crisis Line: dial 988, press 1.",
    points: [
      "Move therapy to phone if an outage is likely; a missed session is a treatment gap.",
      "Buddy checks through a peer partner for veterans with no caregiver on record.",
      "PTSD severity and depression are in the model as main effects and as interactions with hazards.",
      "Every outreach message carries the crisis line, whatever the need it was sent for.",
    ],
    source: "Veterans Crisis Line; VA Whole Health",
    keywords: ["mental", "ptsd", "depression", "crisis", "988", "therapy", "buddy"],
  },
  {
    id: "scam-shield",
    event: "all",
    topic: "outreach",
    title: "The verified message and the scam shield",
    summary: "After disasters, scammers pose as FEMA staff, insurance adjusters and charities. Every Leeward message is built so a veteran can tell it apart.",
    points: [
      "Sent only through VA channels the veteran already uses: VEText, My HealtheVet secure message, the care-team phone. Never a new number.",
      "A four-word verification phrase the veteran can read back to the care team.",
      "The line: the VA will never ask you to pay, wire money, or share bank details.",
      "VSAFE, the VA fraud line: 833-388-7233. Veterans Crisis Line: dial 988, press 1.",
      "No URL shorteners, no phone numbers outside the allow-list, ever.",
    ],
    source: "VA VSAFE; VA News, natural-disaster fraud prevention; FTC disaster scam guidance",
    keywords: ["scam", "fraud", "vsafe", "verify", "verification", "phrase", "vetext", "message", "text"],
  },
  {
    id: "programs-cooling",
    event: "heat",
    topic: "programs",
    title: "Cooling centres, HEAP and rides",
    summary: "NYC opens cooling centres during heat emergencies; HEAP cooling assistance can fund an air conditioner for eligible households. A ride that is booked is an action; a ride that is suggested is a text.",
    points: [
      "Low assets and a transport barrier mean the ride is booked, not suggested; VA transport covers it.",
      "HEAP cooling applications are a five-day-out action, not a during-the-wave one.",
      "The wellness call lands on day two of a wave, when the lag catches up.",
      "271 NYC Parks Cool It! sites are in the reference data as the destination set.",
    ],
    source: "NYC Emergency Management cooling centres; NYS HEAP; VA Beneficiary Travel",
    keywords: ["cooling", "heap", "ride", "transport", "cool it", "air conditioner"],
  },
  {
    id: "programs-partners",
    event: "all",
    topic: "programs",
    title: "One list, many hands: the partner sheet",
    summary: "The care team's list exports with consent flags for partners: wellness checks, rides, housing contacts. Rows without consent are excluded, never redacted.",
    points: [
      "Team Rubicon wellness checks, Combined Arms rides, HUD-VASH housing contacts, all off one list owned by the VA care team.",
      "The outcome log records done, reached, need occurred and partner acknowledgement; twenty to thirty rows a week is what the model can absorb without a retrain.",
      "Every partner action still carries the verification phrase and the never-pay line.",
    ],
    source: "Leeward outreach export; VA community partnership guidance",
    keywords: ["partner", "export", "consent", "team rubicon", "combined arms", "hud-vash", "outcome log"],
  },
];

export function searchLibrary(q: string, event?: EventKind, topic?: Topic): Resource[] {
  const words = q.toLowerCase().split(/\W+/).filter((w) => w.length > 2);
  return LIBRARY.filter((r) => (!event || event === "all" || r.event === event || r.event === "all") && (!topic || r.topic === topic))
    .map((r) => {
      const hay = `${r.title} ${r.summary} ${r.points.join(" ")} ${r.keywords.join(" ")}`.toLowerCase();
      const score = words.reduce((s, w) => s + (r.keywords.some((k) => k.includes(w)) ? 3 : 0) + (r.title.toLowerCase().includes(w) ? 2 : 0) + (hay.includes(w) ? 1 : 0), 0);
      return { r, score };
    })
    .filter((x) => words.length === 0 || x.score > 0)
    .sort((a, b) => b.score - a.score)
    .map((x) => x.r);
}
