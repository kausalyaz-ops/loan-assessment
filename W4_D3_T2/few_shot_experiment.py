"""Reproduce few-shot prompt token estimates and audit illustrative predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
BASE_PROMPT_PATH = ROOT / "prompts" / "zero_shot.txt"
TICKETS_PATH = ROOT / "tickets.csv"
EXAMPLES_PATH = ROOT / "examples.csv"
PREDICTIONS_PATH = ROOT / "predictions.csv"
CONFIG_PATH = ROOT / "prompt_versions.json"

PREDICTION_COLUMNS = {
    "zero_shot": ("zero_shot_category", "zero_shot_priority"),
    "k2": ("k2_category", "k2_priority"),
    "k3": ("k3_category", "k3_priority"),
    "k5": ("k5_category", "k5_priority"),
    "k3_reversed": ("k3_reversed_category", "k3_reversed_priority"),
    "k3_one_wrong_label": ("k3_wrong_category", "k3_wrong_priority"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_example(example: dict[str, str], wrong_label: bool = False) -> str:
    category = "Billing" if wrong_label else example["category"]
    priority = example["priority"]
    return (
        f"Ticket: {example['ticket']}\n"
        f"Answer: {{\"category\":\"{category}\",\"priority\":\"{priority}\"}}"
    )


def make_prompt(
    ticket: dict[str, str],
    selected_ids: list[str],
    examples_by_id: dict[str, dict[str, str]],
    template: str,
    base_prompt: str,
) -> str:
    if not selected_ids:
        return f"{base_prompt}\n\nTicket:\n{ticket['ticket']}"

    rendered_examples: list[str] = []
    for example_id in selected_ids:
        is_wrong_label = example_id.endswith("_WRONG")
        base_id = example_id.removesuffix("_WRONG")
        rendered_examples.append(
            render_example(examples_by_id[base_id], wrong_label=is_wrong_label)
        )

    return (
        template.replace("{examples}", "\n\n".join(rendered_examples))
        .replace("{ticket}", ticket["ticket"])
    )


def score_predictions(
    tickets_by_id: dict[str, dict[str, str]],
    predictions: list[dict[str, str]],
    category_column: str,
    priority_column: str,
) -> tuple[int, int, int]:
    category_correct = 0
    priority_correct = 0
    for prediction in predictions:
        ticket = tickets_by_id[prediction["ticket_id"]]
        category_correct += prediction[category_column] == ticket["true_category"]
        priority_correct += prediction[priority_column] == ticket["true_priority"]
    joint_correct = sum(
        prediction[category_column] == tickets_by_id[prediction["ticket_id"]]["true_category"]
        and prediction[priority_column]
        == tickets_by_id[prediction["ticket_id"]]["true_priority"]
        for prediction in predictions
    )
    return category_correct, priority_correct, joint_correct


def write_dynamic_selections(
    tickets: list[dict[str, str]], examples: list[dict[str, str]]
) -> None:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        norm="l2",
    )
    example_matrix = vectorizer.fit_transform(
        [example["ticket"] for example in examples]
    )
    ticket_matrix = vectorizer.transform([ticket["ticket"] for ticket in tickets])

    rows: list[dict[str, str]] = []
    for ticket, ticket_vector in zip(tickets, ticket_matrix):
        scores = cosine_similarity(ticket_vector, example_matrix).ravel()
        ranked_indexes = sorted(
            range(len(examples)),
            key=lambda index: (-scores[index], examples[index]["example_id"]),
        )[:3]
        for rank, index in enumerate(ranked_indexes, start=1):
            example = examples[index]
            rows.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "rank": str(rank),
                    "example_id": example["example_id"],
                    "similarity": f"{scores[index]:.6f}",
                    "category": example["category"],
                    "priority": example["priority"],
                }
            )

    write_csv(
        ROOT / "dynamic_retrieval.csv",
        ["ticket_id", "rank", "example_id", "similarity", "category", "priority"],
        rows,
    )


def main() -> None:
    base_prompt = BASE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    template = (ROOT / "prompts" / "few_shot_template.txt").read_text(
        encoding="utf-8"
    )
    examples = read_csv(EXAMPLES_PATH)
    examples_by_id = {example["example_id"]: example for example in examples}
    tickets = read_csv(TICKETS_PATH)
    tickets_by_id = {ticket["ticket_id"]: ticket for ticket in tickets}
    predictions = read_csv(PREDICTIONS_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["base_prompt"] != "prompts/zero_shot.txt":
        raise ValueError("The zero-shot prompt path in prompt_versions.json is invalid.")
    if len(tickets) != 10:
        raise ValueError(f"Expected 10 evaluation tickets; found {len(tickets)}.")
    if len(predictions) != len(tickets) or {
        prediction["ticket_id"] for prediction in predictions
    } != set(tickets_by_id):
        raise ValueError("predictions.csv must have exactly one row per evaluation ticket.")

    template_header, separator, _ = template.partition("\n\nFew-shot examples:\n")
    if not separator or template_header.strip() != base_prompt:
        raise ValueError("The few-shot template must preserve the complete zero-shot prompt.")

    experiments = {
        "zero_shot": [],
        **config["versions"],
        **config["challenge_versions"],
    }

    missing_ids = {
        example_id.removesuffix("_WRONG")
        for example_ids in experiments.values()
        for example_id in example_ids
        if example_id.removesuffix("_WRONG") not in examples_by_id
    }
    if missing_ids:
        raise ValueError(f"Unknown example IDs in experiment: {sorted(missing_ids)}")

    encoding = tiktoken.get_encoding("cl100k_base")
    result_rows: list[dict[str, str]] = []
    if set(experiments) != set(PREDICTION_COLUMNS):
        raise ValueError("Prompt versions and prediction columns do not match.")

    for version, selected_ids in experiments.items():
        category_column, priority_column = PREDICTION_COLUMNS[version]
        category_correct, priority_correct, joint_correct = score_predictions(
            tickets_by_id, predictions, category_column, priority_column
        )

        per_ticket_token_counts = [
            len(
                encoding.encode(
                    make_prompt(
                        ticket,
                        selected_ids,
                        examples_by_id,
                        template,
                        base_prompt,
                    )
                )
            )
            for ticket in tickets
        ]
        result_rows.append(
            {
                "version": version,
                "examples": str(len(selected_ids)),
                "accuracy": f"{joint_correct}/{len(tickets)}",
                "correct": str(joint_correct),
                "category_correct": str(category_correct),
                "priority_correct": str(priority_correct),
                "mean_input_tokens": f"{sum(per_ticket_token_counts) / len(tickets):.1f}",
                "tokenizer": "cl100k_base estimate",
                "kind": "illustrative predictions; not model output",
            }
        )

        if version != "zero_shot":
            generated_prompt = template.replace(
                "{examples}",
                "\n\n".join(
                    render_example(
                        examples_by_id[example_id.removesuffix("_WRONG")],
                        wrong_label=example_id.endswith("_WRONG"),
                    )
                    for example_id in selected_ids
                ),
            )
            (ROOT / "prompts" / f"{version}.txt").write_text(
                generated_prompt, encoding="utf-8"
            )

    write_csv(
        ROOT / "experiment_results.csv",
        [
            "version",
            "examples",
            "accuracy",
            "correct",
            "category_correct",
            "priority_correct",
            "mean_input_tokens",
            "tokenizer",
            "kind",
        ],
        result_rows,
    )
    write_dynamic_selections(tickets, examples)

    print("Illustrative predictions only; no LLM inference was run.")
    print("Version             Examples  Accuracy  Mean input tokens/call")
    for row in result_rows:
        print(
            f"{row['version']:<20}{row['examples']:<10}"
            f"{row['accuracy']:<10}{row['mean_input_tokens']}"
        )
    print("Wrote experiment_results.csv and dynamic_retrieval.csv.")


if __name__ == "__main__":
    main()
