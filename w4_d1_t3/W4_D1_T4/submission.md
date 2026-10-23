# W4_D1_T4 — Limitations Lab

This lab records four limitation experiments: a recent-event question,
hallucination checks using a made-up research citation, retrieval of a fact
buried in a long passage, and a multi-step arithmetic problem plus a logic
puzzle. The hallucination experiment also compares the same prompt at
temperature 0 and 1.0.

## Run

In PowerShell, set your API key for the current terminal session and run:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python .\limitations_lab.py
```

The script uses `gpt-4o-mini` by default. To select another compatible model,
set `OPENAI_MODEL` before running it. Do not save an API key in this project.
The script records the actual prompts, model answers, token counts, measured
call times, and your observations in `limitations_results.md`.

For the reasoning check, the expected arithmetic result is **793** and the
coin is in **Box A**. The context-retrieval fact is **Kestrel-47**. For the
recent-event and made-up-reference experiments, inspect the captured answers
and record whether their claims are verifiable; no model response is assumed
in advance.

## Five business-application safeguards

1. Require sources and dates for time-sensitive claims, and flag information
   that cannot be checked against a current trusted source.
2. Require human review before consequential decisions or actions.
3. Validate generated output against authoritative records, schemas, and
   business rules before using it.
4. Minimize prompt context, identify trusted source material, and verify
   extracted facts against that material.
5. Log model/version, prompts, outputs, and approvals with privacy controls;
   monitor errors and provide a safe escalation path.
