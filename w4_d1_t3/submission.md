# W4_D1_T3 — Capabilities Lab

The lab sends one sample prompt each for summarisation, classification,
extraction, translation, and code generation to the OpenAI Chat Completions API.
For each answer, enter a correctness rating from 1 to 5 and an improvement note.
At the end, enter exactly four explanation lines separated by ` | `.

## Run

In PowerShell, set your API key for the current terminal session and run:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python .\capabilities_lab.py
```

The script uses `gpt-4o-mini` by default. To select another compatible model,
set `OPENAI_MODEL` before running it. Do not save an API key in this project.

The completed table, prompt/answer pairs, API-reported input/output token counts,
measured call times, ratings, improvement notes, and four-line explanation are
saved to `capabilities_results.md`.
