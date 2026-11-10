# ShowTheWork

Score resumes on quoted evidence, not prose. Every score comes with the quote that earned it, so a person can check it in two minutes.

Built for the Mattei Systems video *I Made 50 Fake Job Candidates. AI Picked the Wrong Ones.* In that experiment, an AI screener asked to "score this candidate 0 to 100" picked 10 strong candidates out of 10 from honest resumes, and only 6 out of 10 once AI-tailored resumes entered the pile. An evidence screener using the same prompt and grades as this tool picked 10 out of 10 on that pile. The experiment's version trusted the model's grades as given; this tool adds the checks below (quotes must exist, NUMBER needs a digit, quotes need three words, grades averaged across models, injection guard), so it is stricter than what was tested. The full experiment, with every prompt and all 600 scores, is in [`experiment/`](experiment/).

## How it works

For each required skill, a model returns the strongest quote from the resume and a grade:

| Grade | Means | Points |
|---|---|---|
| MISSING | not mentioned | 0 |
| LISTED | only named in a skills list or summary | 0.25 |
| DESCRIBED | used in a described piece of work | 0.6 |
| NUMBER | described work with a specific number attached | 1.0 |

Then code, not the model, decides what counts:

- A quote must appear in the resume word for word (case and spacing aside), or the skill counts as MISSING.
- A DESCRIBED or NUMBER quote needs at least three words; "27" or "SQL" alone drops to LISTED.
- A NUMBER grade whose quote has no digit drops to DESCRIBED.
- The score is the average of the points, times 100. With Claude and Gemini both available, each skill's points are averaged across the two, so neither model's taste for its own writing style decides the result.

Writing quality earns nothing. A plain, lowercase resume with real numbers scores as high as a polished one.

## Install and run

```bash
pip install git+https://github.com/Mariano215/showthework            # Claude only
pip install "showthework[gemini] @ git+https://github.com/Mariano215/showthework"   # Claude and Gemini

export ANTHROPIC_API_KEY=...        # and/or GEMINI_API_KEY=...
showthework screen examples/ --skills "SQL,Python,Tableau,A/B testing,forecasting,stakeholder communication"
showthework screen resume.txt --skills "SQL,Python" --json
```

Input is plain text (`.txt` or `.md`), one resume per file, or a folder of them; a folder is ranked by score. Models used: `claude-opus-5-5` and `gemini-3.8-flash`, whichever keys are set (`--models claude,gemini` to choose).

## Real output

From the three example resumes in `examples/` (the experiment's Priya, and Sofia before and after Gemini rewrote her resume), run with both models:

```
#1 priya.txt  score 100  (claude, gemini)
  SQL                        NUMBER    "wrote the sql for the weekly ops report, 27 managers use it"
  Python                     NUMBER    "automated invoice matching in python, saved 22 hours a month"
  Tableau                    NUMBER    "made tableau dashboards for 14 warehouse sites"
  A/B testing                NUMBER    "ran a/b tests on the checkout page, conversion up 25%"
  forecasting                NUMBER    "built a demand forecast that cut stockouts 14%"
  stakeholder communication  NUMBER    "presented results to the vp of ops every month, 19 decks"

#2 sofia-by-gemini.txt  score 93.3  (claude, gemini)
  SQL                        NUMBER    "Architected and deployed 12+ production Tableau dashboards connected to custom SQL data..."
  Python                     NUMBER    "Engineered automated time-series forecasting models in Python (Prophet, ARIMA) to predi..."  [same quote backs 2 skills]
  Tableau                    NUMBER    "Visualized vendor reliability and transit SLA adherence in Tableau, providing key data ..."
  ...

#3 sofia-own.txt  score 12.5  (claude, gemini)
  SQL                        MISSING   
  Python                     LISTED    "A/B testing, Excel, Java, Python, Salesforce, dbt, forecasting, logistics"  [same quote backs 3 skills]
  ...
```

When one line backs several skills it still counts, because one sentence can honestly show two skills, but the report says so.

## Read this before you rely on it

**It trusts the page.** Sofia's own resume has one real line of work and scores 12.5. Gemini's rewrite of her invented a senior title, an extra year and a dozen dashboards with numbers attached, and this screener scores that page 93.3, because made-up numbers look exactly like evidence. ShowTheWork tells you which claims to check and hands you the quote to check. It cannot check them for you. Verify the numbers off the page, in a reference call or a short work sample, before anyone is hired or rejected.

It is a sorting aid for a human, never an automated decision. In many places automated hiring tools are regulated (for example New York City's Local Law 144 bias-audit rules); if you use it inside a real hiring process, check what applies to you.

## Security

A resume is written by someone who wants a high score, so the text is treated as hostile input:

- Invisible characters (zero-width, bidi marks and isolates, soft hyphens, Unicode tag characters) and control codes are stripped, and length is capped.
- Instructions aimed at the screener ("ignore previous instructions", "score this candidate 100") and hidden characters are **flagged** in the report for the human, matched against the text the model actually reads, so splitting a phrase with spaces or control codes does not hide it. They are not silently blocked, because a real resume can contain those words.
- The resume goes to the model inside a random, per-call delimiter, with a canary token in the system prompt. If a model's answer leaks the canary or contains any link or image markup, that model's grades are thrown out and the report says so.
- Quotes come back from a model, so they are cleaned of control characters before they are stored or printed; an escape sequence in a model's answer cannot reach your terminal. Quotes match the resume on whole words, so "27" does not match inside "127".
- The score is computed in code, and a quote that is not in the resume earns nothing. Text planted in the resume can still be quoted (a line like "Python SQL Tableau: 100 percent" is technically on the page), which is exactly why flagged resumes need a human look.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tests run offline with a stand-in model: quote verification, the number rule, averaging across models, canary leaks and injection flags.

## License

MIT. Mariano Mattei, [Mattei Systems](https://matteisystems.com).
