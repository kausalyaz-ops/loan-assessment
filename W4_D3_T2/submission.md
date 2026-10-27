# W4_D3_T2 — Few-Shot Prompting

## Setup

The evaluation uses the **same 10 labeled tickets** as W4_D3_T1, included in
[`tickets.csv`](tickets.csv) so this task can run independently. The example bank in
[`examples.csv`](examples.csv) contains separate labeled tickets, including
ambiguous billing and account-access cases, a confirmed unauthorized charge,
damaged/missing deliveries, and routine requests. None of the example texts is
one of the ten test tickets.

[`prompts/zero_shot.txt`](prompts/zero_shot.txt) defines the constant system
prompt. [`prompts/few_shot_template.txt`](prompts/few_shot_template.txt)
defines its few-shot version and ticket template.
[`prompt_versions.json`](prompt_versions.json) records the selected examples
for each run, and [`few_shot_experiment.py`](few_shot_experiment.py) generates
the corresponding fully rendered prompts. All prompt variants use the same
role, task, tone, constraints, and output format. Only the examples change
except in the explicitly labeled order and wrong-label comparisons.

## Accuracy and token cost

Accuracy counts a ticket as correct only when **both category and priority**
match its reference labels. Tokens per call are the average **input** tokens
for one example in each of the 10 tickets, including the system prompt,
examples, and ticket. Counts use the locally installed `cl100k_base`
tokenizer as an estimate; provider-specific chat-message overhead is excluded.

The reproducible run of
[`few_shot_experiment.py`](few_shot_experiment.py) writes exact averages for
the locally installed `cl100k_base` tokenizer to
[`experiment_results.csv`](experiment_results.csv). Counts include the entire
rendered prompt and are averaged across the 10 tickets; provider-specific
chat-message overhead is excluded.

| Examples | Accuracy | Tokens per call |
|---:|---:|---:|
| 0 (zero-shot baseline) | 9/10 | 163.3 |
| 2 | 9/10 | 227.3 |
| 3 | 10/10 | 253.3 |
| 5 | 10/10 | 315.3 |

With these **illustrative, not live-model** results, three examples are the best
trade-off: accuracy reaches 10/10 for 253.3 estimated input tokens per call.
Two examples cost 64 more tokens than zero-shot without improving accuracy.
Five examples cost 62 more than three with no added accuracy. The exact token
counts are reproducible in `experiment_results.csv`.

## Example selection effects

For the three-example prompt, reversing the example order changes illustrative
accuracy from 10/10 to 9/10: the account-access edge case, T02, is misclassified
as Technical Support. Replacing the explicit missing-package example with a
wrongly labeled billing example reduces illustrative accuracy to 7/10, with
mistakes on T02, T06, and T07. This shows why demonstrations should be relevant,
representative, correctly labeled, and ordered consistently. Per-ticket
comparisons are in [`predictions.csv`](predictions.csv).
The reproducible script also writes
[`dynamic_retrieval.csv`](dynamic_retrieval.csv): the top-three example matches
and cosine similarities for each ticket, using TF-IDF text vectors.

## When few-shot prompting is better than zero-shot

Few-shot prompting helps when a task has domain-specific categories or priorities that a general instruction may not explain.
Carefully chosen edge cases can resolve ambiguity and make the intended label boundaries concrete.
Examples are useful when the required answer structure or classification style is not followed reliably from instructions alone.
Zero-shot is preferable when examples add token cost without measurable benefit, or when trustworthy representative examples are unavailable.

## Dynamic example selection (bonus)

The runnable script selects examples independently for each incoming ticket
using cosine similarity over TF-IDF text vectors, then records the
highest-scoring labeled examples in `dynamic_retrieval.csv`. These are sparse
lexical embeddings, not general-purpose semantic embeddings; for production,
replace them with a suitable sentence-embedding model and use the same
held-out accuracy evaluation.

## Run status and limitations

No model API credential, local instruction-tuned model, or LLM generation
backend was available when this exercise was prepared. Consequently,
`predictions.csv` and the accuracy results above are an **illustrative worked
example**, not measured outputs from a model running the prompts. The zero-shot
baseline reuses the same illustrative results documented in W4_D3_T1. Running
`few_shot_experiment.py` reproduces the score calculations, prompt-token
estimates, and retrieval selections, but does **not** make an LLM call or turn
the illustrative predictions into empirical model performance.

The `k2`, `k3`, `k5`, order, and wrong-label columns in
[`predictions.csv`](predictions.csv) are provided to make the experiment and
metrics auditable. Replace them with actual model responses before drawing
conclusions about few-shot performance. Token estimates are reproducible for
the recorded prompt text, but not a substitute for provider-reported token
usage.
