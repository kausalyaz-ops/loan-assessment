# W4_D3_T3 — Chain-of-Thought and Sampling

## Experiment setup

This experiment uses the locally cached **GPT-2 base** model. It runs offline
with `transformers`; the model is a base next-token language model, **not an
instruction-tuned assistant**. Results therefore demonstrate generation
settings on an accessible checkpoint and should not be generalized to
instruction-following chat models. The script does not use a remote service or
send prompts outside this machine.

[`word_problems.csv`](word_problems.csv) defines five word problems and their
answer key. The same five problems are each run once with a direct answer
prompt and once with “Think step by step.” The last
standalone number in each generated answer is compared with the answer key;
raw outputs in [`word_problem_responses.csv`](word_problem_responses.csv)
should be checked alongside the automatic score.

The creative-writing comparison generates five samples at each of temperature
0, 0.7, and 1.2. The top-p comparison uses the **same temperature (0.7)** for
five samples at both top-p 0.3 and 0.95. The temperature-0.7, top-p-0.95
samples are reused as the reference condition rather than generated twice,
giving 20 distinct creative generations in total. A fixed seed sequence makes
the local sampling runs reproducible. The exact generated text is recorded in
[`creative_responses.csv`](creative_responses.csv).

## Word problem results

| Prompt | Questions | Correct | Accuracy |
|---|---:|---:|---:|
| Direct answer | 5 | 0 | **0%** |
| “Think step by step” | 5 | 0 | **0%** |

Both prompts received 0/5 from GPT-2. The recorded completions mostly repeat
or echo the question instead of solving it, so asking this base model to think
step by step did not improve accuracy. The answer checker compares the final
standalone generated number with the key; `word_problem_results.csv` contains
the exact scores, and `word_problem_responses.csv` contains each raw output for
manual review.

## Creative prompt and sampling results

| Sampling setting | Runs | Distinct responses | Description |
|---|---:|---:|---|
| Temperature 0 | 5 | 1 | Every run repeated the same story about a girl; it ignored the lighthouse and signal. |
| Temperature 0.7 | 5 | 5 | Every completion differed, but several drifted into generic story-writing advice or descriptions rather than the requested opening. |
| Temperature 1.2 | 5 | 5 | Every completion differed; variation increased, but samples included incoherent or unrelated text. |
| Temperature 0.7, top-p 0.3 | 5 | 2 | Four samples echoed/repeated the prompt; the other described a generic lighthouse. Low variety did not ensure relevance. |
| Temperature 0.7, top-p 0.95 | 5 | 5 | All samples differed, but many drifted into meta-commentary or unrelated narrative instead of following the prompt. |

These are observed differences among the five local completions per setting.
Distinct text is not the same as a relevant or high-quality story. The full
generated samples and computed uniqueness counts are in
[`creative_responses.csv`](creative_responses.csv) and
[`sampling_results.csv`](sampling_results.csv).

## Recommended settings

| Task | Temperature | Top-p | Recommendation |
|---|---:|---:|---|
| Extraction | 0.0 | 1.0 | Use explicit fields, constrain output to the source, and request a fixed schema. |
| Summarisation | 0.2 | 0.95 | Specify audience, length, and required coverage; ground the summary in the source. |
| Brainstorming | 0.9 | 0.95 | State the goal and request diverse, distinct options. |

These are starting points, not universal optima. Select a model-supported
temperature and top-p combination and validate it against the actual task.
[`recommended_settings.csv`](recommended_settings.csv) contains the same
recommendations in machine-readable form.

## When chain-of-thought is a waste of tokens

Step-by-step prompting is wasteful for direct lookups, extraction with an
explicit schema, and simple calculations whose result is readily verified. It
adds latency and tokens without reliably improving correctness. Prefer asking
for a concise final answer or brief check when explanations are not needed.
Measure any reasoning prompt on representative cases; longer reasoning is not
proof that the answer is right.

## Reproduce the experiment

From the task directory, install the packages listed in `requirements.txt` if
needed, then run:

```powershell
python .\run_experiment.py
```

The GPT-2 weights and tokenizer must already be cached locally. The script
refuses to download model files and saves responses and summary tables in this
directory. It does **not** make API calls.
