# The 50-resume experiment

The experiment behind the video. 50 synthetic candidates for one job, each built from facts with a hidden true-fit
score. Their own resumes are rendered from a template (no AI touches them). Claude and Gemini each rewrite every
resume with the prompt every resume tool uses, then Claude and Gemini each score all 150 versions two ways: a naive
"score fit 0 to 100", and an evidence method with the same prompt and grade points as ShowTheWork. The experiment
took the model's grades as given; ShowTheWork adds quote verification, the digit and three-word rules, averaging
across models and the injection guard on top.

```bash
uv run experiment.py selftest          # no API calls
uv run experiment.py resumes           # the 50 template resumes, no API calls
uv run experiment.py run --limit 4     # smoke run
uv run experiment.py run               # full run; cached in out/calls.jsonl, re-runs are free
uv run experiment.py report            # out/summary.json + out/scores.csv
```

Needs `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. Add `--openai-model <id>` for a third writer and screener.

## What is in out/ (the run of 2026-11-10)

| File | Contents |
|---|---|
| `summary.json` | The headline numbers: the shortlists, AI-beats-human rates, invented claims per writer, every candidate's archetype and true fit |
| `scores.json`, `scores.csv` | All 300 version x screener rows, naive and evidence scores |
| `versions.json` | Every resume version: the template original, Claude's rewrite, Gemini's rewrite |
| `calls.jsonl` | Every raw model response, cached by a hash of provider, model and prompt |
| `resumes/` | The 50 template resumes |

## Headline results

- Honest resumes only: both naive screeners shortlisted 10 strong candidates out of 10.
- AI-tailored resumes in the pile: both dropped to 6 of 10; all four wrong picks sent Gemini-tailored resumes.
- Evidence screener on the same pile: 10 of 10 with both models.
- Invented claims: Gemini 50 of 50 rewrites (375 claims), Claude 32 of 50 (52).

One run, one job, synthetic candidates, two models. LLM scores vary run to run. Treat it as a demonstration of the
mechanism, not a census of the market.
