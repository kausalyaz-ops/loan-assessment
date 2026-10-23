"""Run and evaluate five common LLM capabilities using the OpenAI API."""

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
OUTPUT_FILE = Path(__file__).with_name("capabilities_results.md")

TASKS = [
    {
        "name": "Summarisation",
        "prompt": (
            "Summarise the following sample text in one sentence, preserving "
            "its main point:\n\n"
            "A community garden opened on Maple Street in April. Volunteers "
            "built twelve raised beds, and neighbors now grow vegetables and "
            "herbs there. The group donates one third of each harvest to the "
            "local food pantry and holds free gardening lessons every Saturday."
        ),
    },
    {
        "name": "Classification",
        "prompt": (
            "Classify the message as exactly one of: Billing, Delivery, "
            "Technical Support, or Returns. Give a one-sentence reason.\n\n"
            "Sample message: I received the headphones yesterday, but they "
            "will not turn on even after charging them overnight."
        ),
    },
    {
        "name": "Extraction",
        "prompt": (
            "Extract the event details from the sample text as JSON with the "
            "keys event, date, time, location, and fee. Use null for details "
            "that are not stated.\n\n"
            "Sample text: The Riverside Library will host a beginner "
            "watercolor workshop on 14 June at 2:30 p.m. Register at the "
            "front desk. The workshop is free."
        ),
    },
    {
        "name": "Translation",
        "prompt": (
            "Translate the sample text into Spanish. Preserve its polite tone "
            "and meaning.\n\n"
            "Sample text: Please leave the parcel with the building manager "
            "if I am not at home. Thank you for your help."
        ),
    },
    {
        "name": "Code generation",
        "prompt": (
            "Write a small, dependency-free Python function for the sample "
            "requirement. Include a brief docstring and three assertions "
            "covering normal and edge cases.\n\n"
            "Sample requirement: Given a list of temperatures in Celsius, "
            "return a new list containing only values at or above 0, "
            "converted to Fahrenheit using F = C * 9 / 5 + 32. Do not modify "
            "the input list."
        ),
    },
]


def call_model(prompt: str, api_key: str, model: str) -> dict:
    """Return the model answer, API-reported token counts, and elapsed time."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
        raise RuntimeError(f"LLM API returned HTTP {error.code}: {detail}") from error
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
    }


def get_rating() -> int:
    while True:
        value = input("Correctness rating (1-5): ").strip()
        try:
            rating = int(value)
        except ValueError:
            print("Enter a whole number from 1 to 5.")
            continue
        if 1 <= rating <= 5:
            return rating
        print("Enter a whole number from 1 to 5.")


def get_four_line_explanation() -> list[str]:
    while True:
        value = input(
            "Explain best and worst in exactly 4 lines, separated by ' | ': "
        )
        lines = [line.strip() for line in value.split("|")]
        if len(lines) == 4 and all(lines):
            return lines
        print("Provide exactly four non-empty lines separated by ' | '.")


def markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")


def write_report(
    results: list[dict],
    model: str,
    explanation: list[str],
) -> None:
    best = max(results, key=lambda result: result["rating"])
    worst = min(results, key=lambda result: result["rating"])
    rows = [
        "# Capabilities Lab",
        "",
        f"Model: `{model}`",
        "",
        "| Task | Prompt | Answer | Rating | Tokens | Time |",
        "|---|---|---|---:|---|---:|",
    ]
    for result in results:
        tokens = result["tokens"]
        rating = (
            f'{result["rating"]}/5; Improve: '
            f'{result["improvement"]}'
        )
        rows.append(
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in (
                    result["name"],
                    result["prompt"],
                    result["answer"],
                    rating,
                    (
                        f'input {tokens["input"]}, output {tokens["output"]}, '
                        f'total {tokens["total"]}'
                    ),
                    f'{result["elapsed_seconds"]:.2f} s',
                )
            )
            + " |"
        )

    rows.extend(
        [
            "",
            f'Best-rated task: **{best["name"]}** ({best["rating"]}/5).',
            f'Lowest-rated task: **{worst["name"]}** ({worst["rating"]}/5).',
            "",
            "### Best and worst: four-line explanation",
            "",
            *explanation,
            "",
            "Token counts and elapsed time are reported by/measured for the "
            "API call; elapsed time includes network latency.",
            "",
        ]
    )
    OUTPUT_FILE.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set the OPENAI_API_KEY environment variable before running this lab."
        )
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    results = []

    for task in TASKS:
        print(f'\nRunning {task["name"]}...')
        response = call_model(task["prompt"], api_key, model)
        print(f'Answer:\n{response["answer"]}\n')
        rating = get_rating()
        improvement = input("What would you improve in this answer? ").strip()
        if not improvement:
            raise SystemExit("An improvement note is required for every answer.")
        results.append(
            {
                **task,
                **response,
                "rating": rating,
                "improvement": improvement,
            }
        )

    explanation = get_four_line_explanation()
    write_report(results, model, explanation)
    print(f"\nSaved the completed table and evaluation to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
