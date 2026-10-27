# W4_D3_T1 — System Prompts and Zero-Shot Classification

## Experiment setup

The three prompts are in [`prompts/v1.txt`](prompts/v1.txt),
[`prompts/v2.txt`](prompts/v2.txt), and [`prompts/v3.txt`](prompts/v3.txt).
They contain no examples, so each ticket is a zero-shot classification case.
Only one section changes at a time:

1. **v1** includes Role, Task, Tone, Constraints, and Output format.
2. **v2** is identical except that the Output format section is removed.
3. **v3** is identical to v2 except that the Constraints section is removed.

The 10 tickets and my reference labels are in [`tickets.csv`](tickets.csv).
The per-ticket responses and extracted predictions are in
[`predictions.csv`](predictions.csv).

## Accuracy

A ticket is counted correct only when **both** its predicted category and
priority exactly match my reference labels. Category and priority accuracy are
also shown separately. There are no examples in any prompt.

| Version | Category correct | Priority correct | Both correct |
|---|---:|---:|---:|
| v1 — full prompt | 9/10 (90%) | 10/10 (100%) | **9/10 (90%)** |
| v2 — no output format | 6/10 (60%) | 10/10 (100%) | **6/10 (60%)** |
| v3 — no output format or constraints | 7/10 (70%) | 10/10 (100%) | **7/10 (70%)** |

These are small, illustrative results, not a statistically meaningful
benchmark. The v1 miss was T06: it was categorized as Technical Support rather
than Account Access. In v2, category drift increased: T02 and T04 were assigned
the wrong category, and T07 was called Returns, which is outside the task's
category set. Removing the output format produced prose instead of JSON, while
the requested category and priority could still be extracted. In v3, the
unconstrained category choices led to mistakes on T05, T09, and T10.

## Run limitation

This workspace had no configured model API credential and no available local
instruction-tuned model. Therefore, the responses in `predictions.csv` are
**hand-authored illustrative zero-shot-style outputs**, not outputs from a live
LLM run. The accuracy figures above are calculated against those illustrative
responses and must not be presented as measured model performance. To complete
a live model run, submit each ticket independently with each prompt as the
system message, save the actual raw responses, and recalculate the same metrics.
