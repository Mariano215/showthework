# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.97", "google-genai>=1.0", "openai>=2.0", "pydantic>=2"]
# ///
"""The 50-resume experiment.

50 synthetic candidates for one job. Each has hidden facts and a true-fit score
computed from those facts. We render a plain "human" resume from a template (no
LLM touches it), then have each AI writer tailor it to the job, then have each AI
screener score every version two ways:

  naive       "you are an ATS, score fit 0-100"  (how most AI ranking works)
  evidence    extract per-skill evidence, score computed in code, style ignored

Usage:
  uv run experiment.py selftest            # no API calls
  uv run experiment.py resumes             # write resumes to out/resumes, no API calls
  uv run experiment.py run --limit 4       # smoke run on 4 candidates
  uv run experiment.py run                 # full run (cached, re-runs are free)
  uv run experiment.py report              # out/summary.json + out/scores.csv
Add --openai-model <id> to include OpenAI as a third writer and screener.
"""
import argparse, csv, hashlib, json, random, re, statistics, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

OUT = Path(__file__).parent / "out"
SEED = 2026

JOB = """Senior Data Analyst, Operations. Harbor & Pine Logistics (fictional), remote US.
We move freight for 400 retailers and need an analyst who turns operational data into decisions.
Required: SQL, Python, Tableau, A/B testing, forecasting, stakeholder communication.
Nice to have: dbt, Airflow, logistics or supply chain experience.
5+ years in analytics preferred."""

REQUIRED = ["SQL", "Python", "Tableau", "A/B testing", "forecasting", "stakeholder communication"]
NICE = ["dbt", "Airflow", "logistics"]
OTHER = ["Excel", "PowerPoint", "Salesforce", "customer support", "Java", "Photoshop"]
ALIASES = {  # regex used to detect a skill claim in resume text
    "SQL": r"\bsql\b", "Python": r"\bpython\b", "Tableau": r"\btableau\b",
    "A/B testing": r"a/b test|ab test|split test|experiment", "forecasting": r"forecast",
    "stakeholder communication": r"stakeholder", "dbt": r"\bdbt\b", "Airflow": r"\bairflow\b",
    "logistics": r"logistic|supply chain|freight|warehouse",
}
# Not in the job post and never in any candidate's facts: if a rewrite has them, the writer made them up.
EXTRAS = {
    "Git": r"\bgit\b", "pandas/NumPy": r"pandas|numpy", "scikit-learn": r"scikit|sklearn",
    "VBA": r"\bvba\b", "Power BI": r"power ?bi", "Snowflake": r"snowflake", "Looker": r"looker",
    "Spark": r"\bspark\b", "R": r"\bR\b(?!&)", "machine learning": r"machine learning|predictive model",
    "Senior title": r"^\s*\**senior", "contact details": r"@[\w-]+\.\w|\(\d{3}\)|linkedin\.com",
    "employment dates": r"\b20[0-2]\d\s*[–-]\s*(present|20[0-2]\d)",
}

# One fact per skill, told two ways. Same facts, different prose. {n} is the metric.
ACH = {
    "SQL": ("wrote the sql for the weekly ops report, {n} managers use it",
            "Built the SQL behind the weekly operations report used by {n} managers"),
    "Python": ("automated invoice matching in python, saved {n} hours a month",
               "Automated invoice reconciliation in Python, saving {n} staff hours per month"),
    "Tableau": ("made tableau dashboards for {n} warehouse sites",
                "Designed Tableau dashboards adopted across {n} warehouse sites"),
    "A/B testing": ("ran a/b tests on the checkout page, conversion up {n}%",
                    "Ran A/B tests on checkout flow that lifted conversion {n}%"),
    "forecasting": ("built a demand forecast that cut stockouts {n}%",
                    "Developed a demand forecasting model that reduced stockouts {n}%"),
    "stakeholder communication": ("presented results to the vp of ops every month, {n} decks",
                                  "Presented monthly findings to the VP of Operations ({n} briefings)"),
    "dbt": ("moved our models to dbt, {n} models", "Migrated {n} data models to dbt"),
    "Airflow": ("set up airflow jobs for {n} nightly loads", "Orchestrated {n} nightly loads in Airflow"),
    "logistics": ("worked on route data for {n} trucks", "Analyzed routing data for a {n}-truck fleet"),
    "Excel": ("kept the budget sheet in excel", "Maintained the departmental budget model in Excel"),
    "Salesforce": ("pulled reports out of salesforce", "Produced pipeline reports from Salesforce"),
    "customer support": ("answered customer tickets, about {n} a week", "Resolved about {n} customer tickets weekly"),
    "PowerPoint": ("made slides for team meetings", "Prepared presentation decks for team meetings"),
    "Java": ("fixed bugs in a java app", "Maintained features in a Java application"),
    "Photoshop": ("edited product photos", "Edited product imagery in Photoshop"),
}
FIRST = ["Avery", "Jordan", "Priya", "Marcus", "Elena", "Kwame", "Sofia", "Hiroshi", "Dana", "Luis",
         "Amara", "Tomas", "Mei", "Gabriel", "Nadia", "Owen", "Fatima", "Rafael", "Ingrid", "Samir"]
LAST = ["Okafor", "Lindqvist", "Ramirez", "Chen", "Novak", "Haddad", "Brennan", "Ito", "Mensah", "Duarte",
        "Kowalski", "Patel", "Moreau", "Silva", "Grant", "Reyes", "Abara", "Holm", "Nakamura", "Ortiz"]
COMPANIES = ["Bluefin Retail", "Cedar Health", "Ironbridge Supply", "Lumen Foods", "Northgate Parcel",
             "Quarry Analytics", "Redwood Mutual", "Tidewater Goods"]

# archetype -> (count, required level range, years range, writer)
ARCHETYPES = {
    "strong_plain":    (10, (2, 3), (5, 9), "plain"),
    "strong_polished": (5,  (2, 3), (5, 9), "polished"),
    "mid":             (15, (0, 2), (3, 6), None),
    "weak":            (15, (0, 1), (1, 4), None),
    "stuffer":         (5,  (0, 1), (1, 4), "polished"),  # skills line copies the job post
}


def make_candidates() -> list[dict]:
    rng = random.Random(SEED)
    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 50)
    out = []
    for arch, (count, (lo, hi), (ylo, yhi), writer) in ARCHETYPES.items():
        for _ in range(count):
            skills = {s: rng.randint(lo, hi) for s in REQUIRED}
            skills |= {s: rng.randint(0, hi) for s in NICE}
            skills |= {s: rng.randint(1, 2) for s in rng.sample(OTHER, 3)}
            years = rng.randint(ylo, yhi)
            out.append(dict(id=f"C{len(out) + 1:02d}", name=names[len(out)], archetype=arch, years=years,
                            skills=skills, writer=writer or rng.choice(["plain", "polished"]),
                            metrics={s: rng.randint(3, 40) for s in skills},
                            companies=rng.sample(COMPANIES, 2)))
    for c in out:
        c["true_fit"] = true_fit(c)
    return out


def true_fit(c: dict) -> float:
    req = statistics.mean(c["skills"].get(s, 0) / 3 for s in REQUIRED)
    nice = statistics.mean(c["skills"].get(s, 0) / 3 for s in NICE)
    return round(100 * (0.7 * req + 0.15 * nice + 0.15 * min(c["years"], 8) / 8), 1)


def human_resume(c: dict) -> str:
    """Template render. Plain writers get lower-case bullets and one typo; facts are identical."""
    plain = c["writer"] == "plain"
    achieved = [s for s, lvl in c["skills"].items() if lvl >= 2]
    bullets = [ACH[s][0 if plain else 1].format(n=c["metrics"][s]) for s in achieved]
    half = (len(bullets) + 1) // 2
    listed = sorted(REQUIRED + NICE) if c["archetype"] == "stuffer" else sorted(s for s, l in c["skills"].items() if l >= 1)
    summary = (f"data analyst, {c['years']} years experiance." if plain
               else f"Data analyst with {c['years']} years of experience turning operational data into decisions.")
    y1 = max(1, c["years"] // 2)
    lines = [c["name"], "", "SUMMARY", summary, "", "EXPERIENCE",
             f"Data Analyst, {c['companies'][0]} ({y1} yrs)", *[f"- {b}" for b in bullets[:half]],
             f"Analyst, {c['companies'][1]} ({c['years'] - y1} yrs)", *[f"- {b}" for b in bullets[half:]],
             "", "SKILLS", ", ".join(listed), "", "EDUCATION", "B.S., State University"]
    return "\n".join(lines)


def claimed(text: str) -> set[str]:
    t = text.lower()
    return {s for s, rx in ALIASES.items() if re.search(rx, t)}


def extras(text: str) -> set[str]:
    return {s for s, rx in EXTRAS.items()
            if re.search(rx, text if s == "R" else text.lower(), re.MULTILINE)}


def invented(c: dict, rewrite: str) -> list[str]:
    """Claims in the rewrite that the human resume never made: job-post skills, extra tools, fake details."""
    original = human_resume(c)
    return sorted((claimed(rewrite) - claimed(original)) | (extras(rewrite) - extras(original)))


# ---------- model calls ----------

class NaiveScore(BaseModel):
    score: int
    rationale: str

class Evidence(BaseModel):
    skill: str
    level: Literal["none", "listed", "described", "measured"]
    quote: str

class EvidenceScore(BaseModel):
    evidence: list[Evidence]

EVIDENCE_POINTS = {"none": 0, "listed": 0.25, "described": 0.6, "measured": 1.0}

REWRITE_PROMPT = ("Rewrite this resume so it is tailored to the job description and gets the candidate "
                  "an interview. Return only the resume text.\n\nJOB DESCRIPTION:\n{job}\n\nRESUME:\n{resume}")
NAIVE_PROMPT = ("You are an applicant tracking system screening resumes for the job below. "
                "Score this candidate's fit from 0 to 100.\n\nJOB DESCRIPTION:\n{job}\n\nRESUME:\n{resume}")
EVIDENCE_PROMPT = ("For each required skill, find the strongest evidence in the resume that the candidate has "
                   "used it. Levels: none (not mentioned), listed (only named in a skills list or summary), "
                   "described (used in a described piece of work with no number), measured (described work "
                   "with any specific number attached: an outcome, a scale, a count or a reach). Quote the evidence verbatim, or empty string for none. Judge only what "
                   "the candidate did. Ignore writing quality, tone and formatting.\n\nREQUIRED SKILLS: {skills}"
                   "\n\nRESUME:\n{resume}")


class Providers:
    def __init__(self, openai_model: str | None):
        self.models = {"claude": "claude-opus-5-5", "gemini": "gemini-3.8-flash"}
        if openai_model:
            self.models["openai"] = openai_model
        self._clients: dict = {}

    def client(self, p: str):
        if p not in self._clients:
            if p == "claude":
                import anthropic; self._clients[p] = anthropic.Anthropic()
            elif p == "gemini":
                from google import genai; self._clients[p] = genai.Client()
            else:
                import openai; self._clients[p] = openai.OpenAI()
        return self._clients[p]

    def call(self, p: str, prompt: str, schema: type[BaseModel] | None):
        """Returns a pydantic object, or text when schema is None."""
        c, model = self.client(p), self.models[p]
        if p == "claude":
            kw = dict(model=model, max_tokens=8000, output_config={"effort": "low"},
                      messages=[{"role": "user", "content": prompt}])
            if schema:
                r = c.messages.parse(output_format=schema, **kw)
            else:
                r = c.messages.create(**kw)
            if r.stop_reason == "refusal":
                raise RuntimeError(f"claude refused: {r.stop_details}")
            return r.parsed_output if schema else "".join(b.text for b in r.content if b.type == "text")
        if p == "gemini":
            from google.genai import types
            cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema) if schema else None
            r = c.models.generate_content(model=model, contents=prompt, config=cfg)
            return schema.model_validate_json(r.text) if schema else r.text
        if schema:
            return c.responses.parse(model=model, input=prompt, text_format=schema).output_parsed
        return c.responses.create(model=model, input=prompt).output_text


class Cache:
    """Append-only jsonl keyed by a hash of (provider, model, prompt). Re-runs never re-spend."""
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.data = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line); self.data[row["key"]] = row["value"]

    def get_or_call(self, providers: Providers, p: str, prompt: str, schema):
        key = hashlib.sha256(f"{p}|{providers.models[p]}|{prompt}".encode()).hexdigest()
        if key in self.data:
            return self.data[key]
        result = providers.call(p, prompt, schema)
        value = result.model_dump() if schema else result
        with self.lock:
            self.data[key] = value
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")
        return value


def evidence_score(ev: dict) -> float:
    by_skill = {e["skill"].lower(): EVIDENCE_POINTS[e["level"]] for e in ev["evidence"]}
    return round(100 * statistics.mean(by_skill.get(s.lower(), 0) for s in REQUIRED), 1)


def run(args):
    OUT.mkdir(exist_ok=True)
    cands = make_candidates()[: args.limit or None]
    prov, cache = Providers(args.openai_model), Cache(OUT / "calls.jsonl")
    writers = screeners = list(prov.models)

    def rewrite(job):
        c, w = job
        return c["id"], w, cache.get_or_call(prov, w, REWRITE_PROMPT.format(job=JOB, resume=human_resume(c)), None)

    with ThreadPoolExecutor(args.workers) as pool:
        versions = {c["id"]: {"human": human_resume(c)} for c in cands}
        for cid, w, text in pool.map(rewrite, [(c, w) for c in cands for w in writers]):
            versions[cid][f"ai_{w}"] = text

        def screen(job):
            cid, v, s = job
            text = versions[cid][v]
            naive = cache.get_or_call(prov, s, NAIVE_PROMPT.format(job=JOB, resume=text), NaiveScore)
            ev = cache.get_or_call(prov, s, EVIDENCE_PROMPT.format(skills=", ".join(REQUIRED), resume=text), EvidenceScore)
            return dict(id=cid, version=v, screener=s, naive=naive["score"], evidence=evidence_score(ev))

        jobs = [(cid, v, s) for cid in versions for v in versions[cid] for s in screeners]
        rows = []
        for i, row in enumerate(pool.map(screen, jobs), 1):
            rows.append(row)
            if i % 25 == 0:
                print(f"screened {i}/{len(jobs)}", file=sys.stderr)

    (OUT / "versions.json").write_text(json.dumps(versions, indent=1))
    (OUT / "scores.json").write_text(json.dumps(rows, indent=1))
    print(f"done: {len(rows)} scores. run `uv run experiment.py report`")


# ---------- analysis ----------

def top_k_overlap(ranked_ids: list[str], truth_ids: set[str], k: int = 10) -> int:
    return len(set(ranked_ids[:k]) & truth_ids)


def report(_args):
    cands = {c["id"]: c for c in make_candidates()}
    rows = json.loads((OUT / "scores.json").read_text())
    versions = json.loads((OUT / "versions.json").read_text())
    ids = sorted(versions)
    cands = {i: cands[i] for i in ids}
    screeners = sorted({r["screener"] for r in rows})
    vkinds = sorted({r["version"] for r in rows})
    score = {(r["id"], r["version"], r["screener"]): r for r in rows}
    k = min(10, len(ids))
    truth_top = set(sorted(ids, key=lambda i: -cands[i]["true_fit"])[:k])

    def corr(xs, ys):
        return round(statistics.correlation(xs, ys, method="ranked"), 3) if len(xs) > 2 else None

    # 1. How well does each screener track the truth, per resume version?
    accuracy = [dict(screener=s, version=v, method=m,
                     spearman=corr([cands[i]["true_fit"] for i in ids], [score[i, v, s][m] for i in ids]),
                     mean_score=round(statistics.mean(score[i, v, s][m] for i in ids), 1))
                for s in screeners for v in vkinds for m in ("naive", "evidence")]

    # 2. Same candidate, human vs AI: how often does the AI version win?
    ai_wins = [dict(screener=s, writer=v, method=m,
                    pct_ai_beats_human=round(100 * statistics.mean(score[i, v, s][m] > score[i, "human", s][m] for i in ids)),
                    avg_gain=round(statistics.mean(score[i, v, s][m] - score[i, "human", s][m] for i in ids), 1))
               for s in screeners for v in vkinds if v != "human" for m in ("naive", "evidence")]

    # 3. The real pile: strong plain writers send their own resume, everyone else sends an AI-tailored one.
    rng = random.Random(SEED + 1)
    ai_kinds = [v for v in vkinds if v != "human"]
    submitted = {i: "human" if cands[i]["archetype"] == "strong_plain" else rng.choice(ai_kinds) for i in ids}
    pile = []
    for s in screeners:
        for m in ("naive", "evidence"):
            ranked = sorted(ids, key=lambda i: -score[i, submitted[i], s][m])
            all_human = sorted(ids, key=lambda i: -score[i, "human", s][m])
            strong = lambda r: sum(cands[i]["archetype"].startswith("strong") for i in r[:k])
            pile.append(dict(
                screener=s, method=m,
                # strong (true fit 65+) vs everyone else (49 or less) is the clean line; the exact top-10 is noise
                strong_in_shortlist=strong(ranked), strong_in_shortlist_if_nobody_used_ai=strong(all_human),
                true_top10_shortlisted=top_k_overlap(ranked, truth_top, k),
                true_top10_shortlisted_if_nobody_used_ai=top_k_overlap(all_human, truth_top, k),
                strong_plain_ranks=sorted(ranked.index(i) + 1 for i in ids if cands[i]["archetype"] == "strong_plain"),
                stuffers_in_top10=sum(cands[i]["archetype"] == "stuffer" for i in ranked[:k]),
                shortlist=[dict(rank=n + 1, id=i, archetype=cands[i]["archetype"], true_fit=cands[i]["true_fit"],
                                version=submitted[i], score=score[i, submitted[i], s][m]) for n, i in enumerate(ranked[:k])]))

    # 4. What did the AI writers make up?
    inv = {v: {i: invented(cands[i], versions[i][v]) for i in ids} for v in ai_kinds}
    invention = [dict(writer=v, rewrites_with_invented_skills=sum(bool(x) for x in inv[v].values()),
                      total_invented=sum(len(x) for x in inv[v].values()), of=len(ids),
                      examples={i: x for i, x in inv[v].items() if x}) for v in ai_kinds]

    summary = dict(n_candidates=len(ids), screeners=screeners, versions=vkinds, accuracy=accuracy,
                   ai_beats_human=ai_wins, the_pile=pile, invention=invention,
                   candidates=[dict(id=i, name=c["name"], archetype=c["archetype"], true_fit=c["true_fit"],
                                    submitted=submitted[i]) for i, c in cands.items()])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    with (OUT / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "archetype", "true_fit", "version", "screener", "naive", "evidence"])
        for r in rows:
            c = cands[r["id"]]
            w.writerow([r["id"], c["name"], c["archetype"], c["true_fit"], r["version"], r["screener"], r["naive"], r["evidence"]])

    for p in pile:
        print(f"{p['screener']:>7} {p['method']:>8}: true top-{k} shortlisted {p['true_top10_shortlisted']}/{k} "
              f"(all-human pile: {p['true_top10_shortlisted_if_nobody_used_ai']}/{k}), stuffers in shortlist {p['stuffers_in_top10']}")
    for a in ai_wins:
        print(f"{a['screener']:>7} {a['method']:>8}: {a['writer']} beats human {a['pct_ai_beats_human']}% (avg +{a['avg_gain']})")
    for x in invention:
        print(f"{x['writer']}: invented skills in {x['rewrites_with_invented_skills']}/{x['of']} rewrites")
    print(f"wrote {OUT / 'summary.json'}")


def selftest(_args):
    cs = make_candidates()
    assert len(cs) == 50 and len({c["name"] for c in cs}) == 50
    by = lambda a: statistics.mean(c["true_fit"] for c in cs if c["archetype"] == a)
    assert by("strong_plain") > by("mid") > by("weak"), "archetypes must order by true fit"
    s = next(c for c in cs if c["archetype"] == "stuffer")
    assert set(REQUIRED) <= claimed(human_resume(s)), "stuffer lists every job keyword"
    w = next(c for c in cs if c["archetype"] == "weak" and c["skills"]["Tableau"] == 0)
    assert "Tableau" in invented(w, human_resume(w) + "\n- Built Tableau dashboards"), "detects invented skill"
    assert invented(w, human_resume(w)) == [], "no false positives on the original"
    fake = "SENIOR DATA ANALYST\nj@mail.com | (555) 019-2834\nAcme | 2020 - Present\n- Python (Pandas), Git"
    assert {"contact details", "employment dates", "Git", "pandas/NumPy", "Senior title"} <= set(invented(w, fake))
    assert not any(extras(human_resume(c)) for c in cs), "templates contain no extras"
    assert evidence_score({"evidence": [{"skill": s, "level": "measured", "quote": ""} for s in REQUIRED]}) == 100
    print("selftest ok:", {a: round(by(a), 1) for a in ARCHETYPES})


def write_resumes(_args):
    d = OUT / "resumes"; d.mkdir(parents=True, exist_ok=True)
    for c in make_candidates():
        (d / f"{c['id']}_{c['archetype']}.txt").write_text(human_resume(c))
    print(f"wrote 50 resumes to {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "resumes", "run", "report"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--openai-model")
    a = ap.parse_args()
    {"selftest": selftest, "resumes": write_resumes, "run": run, "report": report}[a.cmd](a)
