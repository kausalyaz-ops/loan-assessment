"""Run local GPT-2 word-problem and sampling experiments."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "gpt2"
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    torch.manual_seed(seed)
    sample = temperature > 0

    generation_options: dict[str, object] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sample:
        generation_options["temperature"] = temperature
        generation_options["top_p"] = top_p

    with torch.inference_mode():
        generated = model.generate(**generation_options)

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_last_answer_number(response: str) -> str:
    final_answer = FINAL_ANSWER_PATTERN.search(response)
    if final_answer:
        return final_answer.group(1)

    numbers = NUMBER_PATTERN.findall(response)
    return numbers[-1] if numbers else ""


def score_word_problems(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for mode in ("direct", "think_step_by_step"):
        matching = [row for row in rows if row["prompt_mode"] == mode]
        correct = sum(row["correct"] == "true" for row in matching)
        total = len(matching)
        summaries.append(
            {
                "prompt_mode": mode,
                "correct": str(correct),
                "total": str(total),
                "accuracy": f"{100 * correct / total:.1f}%" if total else "N/A",
            }
        )
    return summaries


def describe_responses(responses: list[str]) -> str:
    if len(set(response.strip().casefold() for response in responses)) == 1:
        return "All five completions were textually identical."
    return (
        f"{len(set(response.strip().casefold() for response in responses))}/5 "
        "completions were textually distinct; inspect the saved text to compare "
        "the story details and coherence."
    )


def main() -> None:
    config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    if config["model"] != MODEL_NAME or config["local_files_only"] is not True:
        raise ValueError("This script only supports the configured offline GPT-2 run.")

    torch.set_num_threads(1)
    model_name = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model.eval()

    problems = read_csv(ROOT / "word_problems.csv")
    if len(problems) != 5:
        raise ValueError(f"Expected exactly five word problems; found {len(problems)}.")

    base_seed = int(config["base_seed"])
    word_responses: list[dict[str, str]] = []
    for prompt_mode in ("direct", "think_step_by_step"):
        for index, problem in enumerate(problems):
            if prompt_mode == "direct":
                prompt = (
                    f"Question: {problem['problem']}\n"
                    "Give a concise answer and show no working.\n"
                    "Answer:"
                )
            else:
                prompt = (
                    f"Question: {problem['problem']}\n"
                    "Think step by step. Finish with a line in this exact form: "
                    "Final answer: <number>.\n"
                    "Solution:"
                )

            response = generate(
                model,
                tokenizer,
                prompt,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=int(config["max_new_tokens_word_problems"]),
                seed=base_seed + index,
            )
            predicted_number = extract_last_answer_number(response)
            word_responses.append(
                {
                    "problem_id": problem["problem_id"],
                    "prompt_mode": prompt_mode,
                    "question": problem["problem"],
                    "expected_answer": problem["expected_answer"],
                    "unit": problem["unit"],
                    "predicted_number": predicted_number,
                    "correct": str(predicted_number == problem["expected_answer"]).lower(),
                    "raw_response": response,
                    "temperature": "0.0",
                    "top_p": "1.0",
                }
            )

    word_fields = [
        "problem_id",
        "prompt_mode",
        "question",
        "expected_answer",
        "unit",
        "predicted_number",
        "correct",
        "temperature",
        "top_p",
        "raw_response",
    ]
    write_csv(ROOT / "word_problem_responses.csv", word_fields, word_responses)
    word_results = score_word_problems(word_responses)
    write_csv(
        ROOT / "word_problem_results.csv",
        ["prompt_mode", "correct", "total", "accuracy"],
        word_results,
    )

    creative_prompt = (ROOT / "creative_prompt.txt").read_text(
        encoding="utf-8"
    ).strip()
    run_count = int(config["creative_runs_per_condition"])
    if run_count != 5:
        raise ValueError("The sampling exercise requires exactly five generations per setting.")

    temperature_conditions = [
        (float(temperature), 0.95)
        for temperature in config["temperature_conditions"]
    ]
    comparison_temperature = float(config["top_p_comparison"]["temperature"])
    top_p_conditions = [
        (comparison_temperature, float(top_p))
        for top_p in config["top_p_comparison"]["top_p_values"]
    ]

    unique_conditions: list[tuple[float, float]] = []
    seen_settings: set[str] = set()
    for temperature, top_p in [*temperature_conditions, *top_p_conditions]:
        setting = f"temperature_{temperature:g}_top_p_{top_p:g}"
        if setting not in seen_settings:
            unique_conditions.append((temperature, top_p))
            seen_settings.add(setting)

    creative_rows: list[dict[str, str]] = []
    creative_by_setting: dict[str, list[str]] = {}
    for temperature, top_p in unique_conditions:
        setting = f"temperature_{temperature:g}_top_p_{top_p:g}"
        creative_by_setting[setting] = []
        for index in range(run_count):
            response = generate(
                model,
                tokenizer,
                creative_prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=int(config["max_new_tokens_creative"]),
                seed=base_seed + 100 + index,
            )
            creative_by_setting[setting].append(response)
            creative_rows.append(
                {
                    "setting": setting,
                    "run": str(index + 1),
                    "seed": str(base_seed + 100 + index),
                    "temperature": f"{temperature:g}",
                    "top_p": f"{top_p:g}",
                    "prompt": creative_prompt,
                    "response": response,
                }
            )

    write_csv(
        ROOT / "creative_responses.csv",
        ["setting", "run", "seed", "temperature", "top_p", "prompt", "response"],
        creative_rows,
    )

    sampling_results: list[dict[str, str]] = []
    for temperature, top_p in unique_conditions:
        setting = f"temperature_{temperature:g}_top_p_{top_p:g}"
        responses = creative_by_setting[setting]
        sampling_results.append(
            {
                "setting": setting,
                "runs": str(len(responses)),
                "distinct_responses": str(
                    len(set(response.strip().casefold() for response in responses))
                ),
                "description": describe_responses(responses),
            }
        )

    write_csv(
        ROOT / "sampling_results.csv",
        ["setting", "runs", "distinct_responses", "description"],
        sampling_results,
    )

    print(f"Model: {model_name} (offline, base model; not instruction-tuned)")
    print("Word problem accuracy (final-number checker; manually review raw responses):")
    for result in word_results:
        print(
            f"  {result['prompt_mode']}: "
            f"{result['correct']}/{result['total']} ({result['accuracy']})"
        )
    print("Creative generations: five for each temperature and top-p setting.")
    for result in sampling_results:
        print(
            f"  {result['setting']}: "
            f"{result['distinct_responses']}/5 distinct responses"
        )
    print("Saved raw generations and experiment tables in the task directory.")


if __name__ == "__main__":
    main()
