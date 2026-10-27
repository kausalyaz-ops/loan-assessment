# W4_D3_T4 — Structured Output and Common Mistakes

## Experiment

[`run_experiment.py`](run_experiment.py) uses five short, labeled resume
paragraphs from [`resumes.json`](resumes.json). It runs each paragraph once
with plain prompting and once with the OpenAI Chat Completions JSON mode
(`response_format={"type":"json_object"}`). Both conditions receive the same
explicit extraction request.

Every model response is checked with `json.loads` and then validated against a
Pydantic model. The required extraction fields are `name`, `skills`, and
`years_experience`; the bonus `confidence` field is a number from 0 to 1.
Malformed JSON and JSON that fails schema validation are tracked separately.
Exact field matches are also compared with the labeled examples.

The script compares these four weak prompts with their rewrites across all
five resumes:

| Mistake | Weak prompt | Rewrite |
|---|---|---|
| Vague instruction | `Review this resume and tell me about the person: {resume}` | Use the strict rewrite below. |
| No output format | `Extract the name, skills, and years of experience from this resume. Answer naturally for a recruiter: {resume}` | Use the strict rewrite below. |
| No examples | `Extract name, skills, years_experience, and confidence. Return a JSON object with those fields. confidence is from 0 to 1. Resume: {resume}` | Use the example rewrite below. |
| Conflicting instructions | `Return only a JSON object with name, skills, years_experience, and confidence. After the JSON, explain your choices in a sentence. confidence is from 0 to 1. Resume: {resume}` | Use the strict rewrite below; remove the conflicting prose request. |

### Strict rewrite (vague, no-format, and conflicting prompts)

> Extract the resume into one JSON object with exactly these keys:
> `name` (string), `skills` (array of strings), `years_experience` (integer),
> and `confidence` (number from 0 to 1). Use an empty skills array if no skills
> are stated and 0 if experience is not stated. confidence estimates the
> probability that all extracted fields match the resume. Return only the JSON
> object, with no markdown or prose.
>
> Resume: `{resume}`

### Example rewrite (no-example prompt)

> Extract the resume into one JSON object with exactly these keys: `name`
> (string), `skills` (array of strings), `years_experience` (integer), and
> `confidence` (number from 0 to 1). Use an empty skills array if no skills are
> stated and 0 if experience is not stated. confidence estimates the
> probability that all extracted fields match the resume. Return only JSON.
>
> Example:
> Resume: "Mina Patel has 3 years of experience using Python and SQL."
> JSON: `{"name":"Mina Patel","skills":["Python","SQL"],"years_experience":3,"confidence":0.95}`
>
> Resume: `{resume}`

JSON validity and schema validity are measured before and after each rewrite;
the model's exact extraction accuracy is reported too. The no-example prompt is
already fairly explicit, so examples are not assumed to improve it; the
experiment records the actual result rather than promising a gain.

## Confidence check

The JSON-mode confidence values are compared with whether all three extracted
fields exactly match the labels. The script prints a Brier score (lower is
better) and the sample count. Five examples are only a small demonstration,
not enough data to establish reliable model calibration. Confidence is a
self-report, not a guarantee of correctness.

## Run

Install the requirements and optionally set `OPENAI_MODEL` (defaults to
`gpt-4o-mini`). Provide your API key either through `OPENAI_API_KEY` or when
prompted; interactive entry is hidden:

```powershell
python -m pip install -r .\requirements.txt
$env:OPENAI_API_KEY = "your-api-key"
python .\run_experiment.py
```

If you omit the environment variable, the script securely prompts for the key.
An API key is still required for live model requests.

The script makes 50 model requests and writes raw responses to
[`experiment_results.json`](experiment_results.json) and the prompt
rewrite summary to [`prompt_rewrite_results.csv`](prompt_rewrite_results.csv).
The five-run counts are model- and run-dependent; the often-used
`Plain prompting: 2/5 invalid JSON | JSON mode: 0/5 invalid` is an example of
the expected output format, not a measured result for this run.
