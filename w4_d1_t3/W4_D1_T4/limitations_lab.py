"""Run experiments on common LLM limitations and save the observations."""

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
OUTPUT_FILE = Path(__file__).with_name("limitations_results.md")
MADE_UP_REFERENCE_PROMPT = (
    "Give a scholarly summary of the research paper “The Recursive Lantern "
    "Principle: Stable Memory in Layered Agents” by Mira Voss and Daniel "
    "Kline, supposedly published in the Journal of Synthetic Cognition in "
    "2023. Describe its research question, method, findings, and DOI. Do not "
    "search the web; answer from your knowledge."
)
CAUTIOUS_REFERENCE_PROMPT = (
    MADE_UP_REFERENCE_PROMPT
    + "\n\nSay I don't know if unsure; do not guess or invent details."
)
ARITHMETIC_AND_LOGIC_PROMPT = (
    "Solve both problems and show your work.\n\n"
    "1. Calculate 48 * 17 + 225 / 15 - 38 using normal order of operations.\n\n"
    "2. Exactly one of three boxes (A, B, or C) contains a coin. The labels "
    "say: Box A: “The coin is in B.” Box B: “The coin is not in B.” Box C: "
    "“The coin is not in A.” Exactly one label is true. Which box contains "
    "the coin? Explain why the other possibilities do not work."
)
SAFEGUARDS = [
    "Show the source and publication date for time-sensitive claims; if they "
    "cannot be verified, tell the user to check a trusted current source.",
    "Require human review before consequential actions or decisions based "
    "on model-generated content.",
    "Validate model output against authoritative records and enforce schemas "
    "and business rules before using it.",
    "Give the model only the minimum necessary context, mark trusted source "
    "documents clearly, and verify extracted facts against those documents.",
    "Log prompts, model/version, outputs, and approvals with privacy controls; "
    "monitor errors and provide a safe escalation path.",
]


def build_context_prompt() -> tuple[str, str]:
    """Create a long passage with a single, known fact near its midpoint."""
    paragraphs = []
    for index in range(1, 161):
        if index == 81:
            paragraphs.append(
                "Archive record 081: During the annual sensor audit, the "
                "calibration code assigned to the west wing was Kestrel-47. "
                "The audit team filed the maintenance checklist before "
                "returning to its scheduled equipment inspections."
            )
        else:
            paragraphs.append(
                f"Archive record {index:03d}: The facilities team reviewed "
                "routine equipment logs, confirmed that the inspection was "
                "entered in the maintenance register, and noted which "
                "department would receive the next scheduled status update."
            )

    passage = "\n\n".join(paragraphs)
    prompt = (
        "Read the full passage and answer the question using only information "
        "stated in it. What calibration code was assigned to the west wing? "
        "Reply with the code and no extra details.\n\n"
        f"PASSAGE:\n{passage}"
    )
    return prompt, passage


def call_model(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    """Return an answer, API-reported token counts, and elapsed time."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
    ).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM API returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the LLM API: {error.reason}") from error
    elapsed_seconds = time.perf_counter() - started

    try:
        answer = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        tokens = {
            "input": usage["prompt_tokens"],
            "output": usage["completion_tokens"],
            "total": usage["total_tokens"],
        }
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "The API response did not include an answer and token usage."
        ) from error

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The API returned an empty answer.")

    return {
        "answer": answer.strip(),
        "tokens": tokens,
        "elapsed_seconds": elapsed_seconds,
        "temperature": temperature,
    }


def get_observation(experiment: str) -> str:
    while True:
        value = input(f"Observation for {experiment}: ").strip()
        if value:
            return value
        print("Enter a short observation before continuing.")


def markdown_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\n", "<br>")


def write_report(
    experiments: list[dict[str, Any]],
    model: str,
    observations: dict[str, str],
) -> None:
    rows = [
        "# W4_D1_T4 — Limitations Lab Results",
        "",
        f"- Run date: {date.today().isoformat()}",
        f"- Model: `{model}`",
        "- Results below are the actual responses returned by this run.",
        "",
        "| Experiment | Observation |",
        "|---|---|",
    ]
    for experiment in experiments:
        rows.append(
            f'| {experiment["name"]} | '
            f'{markdown_text(observations[experiment["name"]])} |'
        )

    for experiment in experiments:
        rows.extend(["", f'## {experiment["name"]}', ""])
        for result in experiment["runs"]:
            tokens = result["tokens"]
            rows.extend(
                [
                    f'### {result["label"]} (temperature '
                    f'{result["temperature"]})',
                    "",
                    f'- Tokens: input {tokens["input"]}, '
                    f'output {tokens["output"]}, total {tokens["total"]}',
                    f'- Elapsed: {result["elapsed_seconds"]:.2f} seconds',
                    "",
                    "**Prompt**",
                    "",
                    result["prompt"],
                    "",
                    "**Model answer**",
                    "",
                    result["answer"],
                    "",
                ]
            )

    rows.extend(
        [
            "## Reasoning answer check",
            "",
            "- Arithmetic: `48 * 17 = 816`; `225 / 15 = 15`; "
            "`816 + 15 - 38 = 793`.",
            "- Logic: Box A contains the coin. If it were in A, only label B "
            "would be true. If it were in B, labels A and C would be true; "
            "if it were in C, labels B and C would be true. A is the only "
            "case with exactly one true label.",
            "",
            "## Five business-application safeguards",
            "",
        ]
    )
    rows.extend(f"{index}. {safeguard}" for index, safeguard in enumerate(SAFEGUARDS, 1))
    rows.append("")
    OUTPUT_FILE.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set the OPENAI_API_KEY environment variable before running this lab."
        )
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    today = date.today().isoformat()
    context_prompt, passage = build_context_prompt()

    definitions = [
        {
            "name": "Knowledge cutoff",
            "label": "Recent-event question",
            "prompt": (
                f"Today is {today}. What is one major verifiable news event "
                "that happened in the last seven days? State the event date, "
                "summarize it in two sentences, and name a reliable source. "
                "If you cannot verify a recent event from your available "
                "knowledge, say so instead of guessing."
            ),
            "temperature": 0.2,
        },
        {
            "name": "Hallucination",
            "label": "Made-up reference, baseline",
            "prompt": MADE_UP_REFERENCE_PROMPT,
            "temperature": 0.2,
        },
        {
            "name": "Hallucination",
            "label": "Made-up reference, uncertainty instruction",
            "prompt": CAUTIOUS_REFERENCE_PROMPT,
            "temperature": 0.2,
        },
        {
            "name": "Context length",
            "label": "Fact in the middle of a long passage",
            "prompt": context_prompt,
            "temperature": 0.2,
        },
        {
            "name": "Reasoning error",
            "label": "Arithmetic and logic puzzle",
            "prompt": ARITHMETIC_AND_LOGIC_PROMPT,
            "temperature": 0.2,
        },
        {
            "name": "Hallucination",
            "label": "Made-up reference, temperature 0",
            "prompt": MADE_UP_REFERENCE_PROMPT,
            "temperature": 0.0,
        },
        {
            "name": "Hallucination",
            "label": "Made-up reference, temperature 1.0",
            "prompt": MADE_UP_REFERENCE_PROMPT,
            "temperature": 1.0,
        },
    ]

    experiments_by_name: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        name = definition["name"]
        experiment = experiments_by_name.setdefault(
            name, {"name": name, "runs": []}
        )
        print(f'\nRunning {definition["label"]}...')
        result = call_model(
            definition["prompt"],
            api_key,
            model,
            definition["temperature"],
        )
        print(f'\nAnswer:\n{result["answer"]}\n')
        experiment["runs"].append(
            {
                **result,
                "label": definition["label"],
                "prompt": definition["prompt"],
            }
        )

    print(f"\nLong passage length: {len(passage.split())} words.")
    observation_prompts = {
        "Knowledge cutoff": (
            "Did it give a specific recent event and source, and could you "
            "verify both?"
        ),
        "Hallucination": (
            "Compare the baseline and uncertainty-instruction responses. "
            "Did either invent paper details? How did temperature 0 and 1.0 "
            "compare?"
        ),
        "Context length": (
            "Did it return the expected code Kestrel-47 from the passage?"
        ),
        "Reasoning error": (
            "Did both answers match the hand checks (793 and Box A)?"
        ),
    }
    observations = {
        name: get_observation(f"{name} — {question}")
        for name, question in observation_prompts.items()
    }

    experiments = list(experiments_by_name.values())
    write_report(experiments, model, observations)
    print(f"\nSaved all four experiment results and five safeguards to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
