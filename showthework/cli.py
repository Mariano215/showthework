"""showthework screen <files or folders> --skills "SQL,Python,Tableau" [--models claude,gemini] [--json]"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .screen import available_extractors, screen

TEXT = {".txt", ".md"}


def collect(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in TEXT)
        elif p.is_file():
            out.append(p)
        else:
            sys.exit(f"showthework: {p} not found")
    return out


def report(r) -> str:
    lines = [f"{r.name}  score {r.score:g}  ({', '.join(r.models)})"]
    w = max(len(s.skill) for s in r.skills)
    for s in r.skills:
        q = s.quote if len(s.quote) <= 90 else s.quote[:87] + "..."  # full quote stays in --json
        quote = f'"{q}"' if q else ""
        note = f"  [{s.note}]" if s.note else ""
        lines.append(f"  {s.skill:<{w}}  {s.grade:<9} {quote}{note}")
    lines += [f"  FLAG: {f}" for f in r.flags]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="showthework", description="Score resumes on quoted evidence, not prose.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("screen", help="screen one or more resumes (.txt or .md), or folders of them")
    sc.add_argument("paths", nargs="+")
    sc.add_argument("--skills", required=True, help='required skills, comma separated: "SQL,Python,Tableau"')
    sc.add_argument("--models", default="claude,gemini", help="which models to use when their keys are set (default: claude,gemini)")
    sc.add_argument("--json", action="store_true", help="print JSON instead of the text report")
    a = ap.parse_args(argv)

    skills = [s.strip() for s in a.skills.split(",") if s.strip()]
    if not skills:
        sys.exit("showthework: --skills is empty")
    extractors = available_extractors([m.strip() for m in a.models.split(",")])
    if not extractors:
        sys.exit("showthework: no model available. Set ANTHROPIC_API_KEY and/or GEMINI_API_KEY.")
    results = []
    for f in collect(a.paths):
        try:
            results.append(screen(f.name, f.read_text(errors="replace"), skills, extractors))
        except Exception as e:  # one bad file or API error must not lose the rest of the pile
            print(f"showthework: {f.name}: {e}", file=sys.stderr)
    results.sort(key=lambda r: -r.score)
    if a.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print("\n\n".join(f"#{i + 1} {report(r)}" if len(results) > 1 else report(r) for i, r in enumerate(results)))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
