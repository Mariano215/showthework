"""Score a resume on evidence, not prose.

For every required skill a model returns the strongest quote from the resume and a grade:
MISSING (not mentioned), LISTED (only named), DESCRIBED (used in described work), NUMBER (described work with a
specific number). Code then checks each quote appears in the resume word for word; a quote that is not there counts
as MISSING, and a NUMBER quote without a digit drops to DESCRIBED. The score is arithmetic over those grades,
averaged across models, so writing style earns nothing and an invented quote earns nothing. Text planted in the
resume itself can still be quoted, which is why guard.flags() surfaces instruction-like text for the human.
"""
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel

from . import guard

POINTS = {"MISSING": 0.0, "LISTED": 0.25, "DESCRIBED": 0.6, "NUMBER": 1.0}
PROMPT = (
    "You check resumes for evidence. For each required skill, find the strongest evidence in the resume that the "
    "candidate has used it, and grade it: MISSING (not mentioned), LISTED (only named in a skills list or summary), "
    "DESCRIBED (used in a described piece of work with no number), NUMBER (described work with any specific number "
    "attached: an outcome, a scale, a count or a reach). Quote the evidence exactly as written in the resume, or an "
    "empty string for MISSING. Judge only what the candidate did. Ignore writing quality, tone and formatting."
)
CLAUDE_MODEL = "claude-opus-5-5"
GEMINI_MODEL = "gemini-3.8-flash"


class Evidence(BaseModel):
    skill: str
    grade: Literal["MISSING", "LISTED", "DESCRIBED", "NUMBER"]
    quote: str


class Extraction(BaseModel):
    evidence: list[Evidence]


@dataclass
class SkillResult:
    skill: str
    grade: str
    quote: str
    points: float
    note: str = ""


@dataclass
class Result:
    name: str
    score: float
    skills: list[SkillResult]
    models: list[str]
    flags: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%$.+/@-]+", " ", s.lower())).strip()


def verify(e: Evidence, resume: str) -> tuple[str, str]:
    """The grade the quote can actually support, and why it changed."""
    if e.grade == "MISSING":
        return "MISSING", ""
    # Whole-word match: "27" must not match inside "127". The skill name is deliberately not required in the quote:
    # "presented results to the VP every month" is real stakeholder evidence that never says "stakeholder".
    if not e.quote.strip() or f" {_norm(e.quote)} " not in f" {_norm(resume)} ":
        return "MISSING", "quote not found in the resume"
    if e.grade in ("DESCRIBED", "NUMBER") and len(_norm(e.quote).split()) < 3:  # "27" or "SQL" alone is not described work
        return "LISTED", "quote too short to show described work"
    if e.grade == "NUMBER" and not re.search(r"\d", e.quote):
        return "DESCRIBED", "no number in the quote"
    return e.grade, ""


Extractor = Callable[[str, str, list[str]], tuple[Extraction, str]]  # (system, user, skills) -> (parsed, raw text)


def claude_extractor() -> Extractor:
    import anthropic
    client = anthropic.Anthropic()

    def run(system, user, skills):
        r = client.messages.parse(
            model=CLAUDE_MODEL, max_tokens=8000, system=system, output_config={"effort": "low"}, output_format=Extraction,
            messages=[{"role": "user", "content": f"REQUIRED SKILLS: {', '.join(skills)}\n\n{user}"}],
        )
        if r.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined this resume ({r.stop_details})")
        raw = "".join(getattr(b, "text", "") for b in r.content)
        return r.parsed_output, raw
    return run


def gemini_extractor() -> Extractor:
    from google import genai
    from google.genai import types
    client = genai.Client()

    def run(system, user, skills):
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=f"REQUIRED SKILLS: {', '.join(skills)}\n\n{user}",
            config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json", response_schema=Extraction),
        )
        return Extraction.model_validate_json(r.text), r.text
    return run


def available_extractors(names: list[str]) -> dict[str, Extractor]:
    out = {}
    for n in names:
        if n == "claude" and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            out[n] = claude_extractor()
        elif n == "gemini" and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            try:  # a Gemini key without the optional package must not crash a Claude-only install
                out[n] = gemini_extractor()
            except ImportError:
                print('showthework: GEMINI_API_KEY is set but google-genai is not installed; using Claude only '
                      '(pip install "showthework[gemini]")', file=sys.stderr)
    return out


def screen(name: str, raw_resume: str, skills: list[str], extractors: dict[str, Extractor]) -> Result:
    if not extractors:
        raise RuntimeError("No model available: set ANTHROPIC_API_KEY and/or GEMINI_API_KEY.")
    found_flags = guard.flags(raw_resume)
    resume = guard.sanitize(raw_resume)
    per_skill: dict[str, list[SkillResult]] = {s: [] for s in skills}
    used = []
    for model, extract in extractors.items():
        system, user, canary = guard.wrap(PROMPT, resume)
        parsed, raw = extract(system, user, skills)
        problems = guard.output_problems(raw, canary)
        if problems:  # a compromised answer is not evidence: report it and ignore this model's grades
            found_flags += [f"{model}: {p}" for p in problems]
            continue
        used.append(model)
        by_skill = {e.skill.strip().lower(): e for e in parsed.evidence}
        for s in skills:
            e = by_skill.get(s.lower()) or Evidence(skill=s, grade="MISSING", quote="")
            grade, note = verify(e, resume)
            # Keep only the sanitized quote: it is model output and gets printed to a terminal.
            quote = re.sub(r"\s+", " ", guard.sanitize(e.quote)) if grade != "MISSING" else ""
            per_skill[s].append(SkillResult(s, grade, quote, POINTS[grade], note))
    if not used:
        raise RuntimeError(f"Every model's answer failed validation for {name}: {found_flags}")
    skills_out = []
    for s, rs in per_skill.items():
        best = max(rs, key=lambda r: r.points)  # show the best supported quote; the score uses the average
        notes = sorted({r.note for r in rs if r.note})
        skills_out.append(SkillResult(s, best.grade, best.quote, sum(r.points for r in rs) / len(rs), "; ".join(notes)))
    # One line can honestly evidence two skills ("SQL models behind Tableau dashboards"), so it is allowed, but shown.
    seen: dict[str, int] = {}
    for r in skills_out:
        if r.quote:
            seen[_norm(r.quote)] = seen.get(_norm(r.quote), 0) + 1
    for r in skills_out:
        if r.quote and seen[_norm(r.quote)] > 1:
            r.note = "; ".join(x for x in [r.note, f"same quote backs {seen[_norm(r.quote)]} skills"] if x)
    score = round(100 * sum(r.points for r in skills_out) / len(skills_out), 1)
    return Result(name, score, skills_out, used, found_flags)
