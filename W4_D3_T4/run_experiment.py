"""Compare prompt-only and JSON-mode resume extraction."""

from __future__ import annotations

import csv
import getpass
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from openai import OpenAI


ROOT = Path(__file__).resolve().parent
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RUNS_PER_CONDITION = 5


class ResumeExtraction(BaseModel):
    """The structured fields extracted from a resume."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    skills: list[str]
    years_experience: int = Field(ge=0, le=60)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class ValidationResult:
    parsed: ResumeExtraction | None
    valid_json: bool
    error: str


SYSTEM_PROMPT = (
    "You are a careful resume parser. Extract only information stated in the "
    "resume. Do not infer missing facts."
)

STRICT_PROMPT = """Extract the resume into one JSON object with exactly these keys:
{{
  "name": string,
  "skills": array of strings,
  "years_experience": integer,
  "confidence": number from 0 to 1
}}
Use an empty skills array if no skills are stated. Use 0 for years_experience
if it is not stated. confidence estimates the probability that all extracted
fields match the resume. Return only the JSON object, with no markdown or prose.

Resume:
{resume}"""

EXAMPLE_PROMPT = """Extract the resume into one JSON object with exactly these keys:
name (string), skills (array of strings), years_experience (integer), and
confidence (number from 0 to 1). Use an empty skills array if no skills are
stated and 0 if experience is not stated. confidence estimates the probability
that all extracted fields match the resume. Return only JSON.

Example:
Resume: "Mina Patel has 3 years of experience using Python and SQL."
JSON: {{"name":"Mina Patel","skills":["Python","SQL"],"years_experience":3,"confidence":0.95}}

Resume:
{resume}"""

BAD_PROMPTS: dict[str, str] = {
    "vague_instruction": (
        "Review this resume and tell me about the person:\n{resume}"
    ),
    "no_output_format": (
        "Extract the name, skills, and years of experience from this resume. "
        "Answer naturally for a recruiter:\n{resume}"
    ),
    "no_examples": (
        "Extract name, skills, years_experience, and confidence. Return a JSON "
        "object with those fields. confidence is from 0 to 1.\nResume:\n{resume}"
    ),
    "conflicting_instructions": (
        "Return only a JSON object with name, skills, years_experience, and "
        "confidence. After the JSON, explain your choices in a sentence. "
        "confidence is from 0 to 1.\nResume:\n{resume}"
    ),
}

REWRITTEN_PROMPTS: dict[str, str] = {
    "vague_instruction": STRICT_PROMPT,
    "no_output_format": STRICT_PROMPT,
    "no_examples": EXAMPLE_PROMPT,
    "conflicting_instructions": STRICT_PROMPT,
}


def load_resumes() -> list[dict[str, Any]]:
    with (ROOT / "resumes.json").open(encoding="utf-8") as file:
        resumes = json.load(file)
    if len(resumes) != RUNS_PER_CONDITION:
        raise ValueError(f"Expected exactly {RUNS_PER_CONDITION} resumes.")
    return resumes


def validate_output(raw: str) -> ValidationResult:
    """Parse every response as JSON, then validate the exact extraction schema."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return ValidationResult(None, False, f"Invalid JSON: {error.msg}")

    try:
        parsed = ResumeExtraction.model_validate(value)
    except ValidationError as error:
        return ValidationResult(None, True, str(error))
    return ValidationResult(parsed, True, "")


def ask_model(
    client: OpenAI,
    prompt: str,
    *,
    json_mode: bool,
) -> str:
    options: dict[str, Any] = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**options)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The model returned an empty response.")
    return content


def create_record(
    client: OpenAI,
    *,
    resume: dict[str, Any],
    condition: str,
    prompt: str,
    json_mode: bool,
) -> dict[str, Any]:
    raw = ask_model(client, prompt, json_mode=json_mode)
    validation = validate_output(raw)
    return {
        "resume_id": resume["id"],
        "condition": condition,
        "prompt": prompt,
        "raw_output": raw,
        "valid_json": validation.valid_json,
        "schema_valid": validation.parsed is not None,
        "validation_error": validation.error,
        "parsed": (
            validation.parsed.model_dump() if validation.parsed is not None else None
        ),
        "expected": {
            "name": resume["expected"]["name"],
            "skills": resume["expected"]["skills"],
            "years_experience": resume["expected"]["years_experience"],
        },
    }


def extraction_is_correct(record: dict[str, Any]) -> bool:
    parsed = record["parsed"]
    expected = record["expected"]
    if parsed is None:
        return False
    return (
        parsed["name"].strip().casefold() == expected["name"].strip().casefold()
        and {skill.casefold() for skill in parsed["skills"]}
        == {skill.casefold() for skill in expected["skills"]}
        and parsed["years_experience"] == expected["years_experience"]
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(records),
        "invalid_json": sum(not record["valid_json"] for record in records),
        "schema_valid": sum(record["schema_valid"] for record in records),
        "correct_extractions": sum(extraction_is_correct(record) for record in records),
    }


def confidence_brier_score(records: list[dict[str, Any]]) -> float | None:
    evaluated = [
        record
        for record in records
        if record["parsed"] is not None
    ]
    if not evaluated:
        return None
    return sum(
        (record["parsed"]["confidence"] - float(extraction_is_correct(record))) ** 2
        for record in evaluated
    ) / len(evaluated)


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("OpenAI API key (input hidden): ").strip()
    if not api_key:
        raise RuntimeError(
            "An API key is required. Set OPENAI_API_KEY or enter a key when prompted."
        )
    return api_key


def write_results(records: list[dict[str, Any]]) -> None:
    (ROOT / "experiment_results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "prompt_type",
        "version",
        "total",
        "invalid_json",
        "schema_valid",
        "correct_extractions",
    ]
    summary_rows: list[dict[str, Any]] = []
    for prompt_type in sorted({record["prompt_type"] for record in records}):
        for version in ("original", "rewritten"):
            matching = [
                record
                for record in records
                if record["prompt_type"] == prompt_type
                and record["version"] == version
            ]
            summary_rows.append(
                {"prompt_type": prompt_type, "version": version, **summarize(matching)}
            )
    with (ROOT / "prompt_rewrite_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    from openai import OpenAI

    api_key = get_api_key()
    resumes = load_resumes()
    client = OpenAI(api_key=api_key)
    records: list[dict[str, Any]] = []

    for json_mode in (False, True):
        condition = "json_mode" if json_mode else "plain_prompting"
        for resume in resumes:
            prompt = STRICT_PROMPT.format(resume=resume["text"])
            record = create_record(
                client,
                resume=resume,
                condition=condition,
                prompt=prompt,
                json_mode=json_mode,
            )
            record["prompt_type"] = "format_comparison"
            record["version"] = condition
            records.append(record)

    for prompt_type in BAD_PROMPTS:
        for version, templates in (
            ("original", BAD_PROMPTS),
            ("rewritten", REWRITTEN_PROMPTS),
        ):
            template = templates[prompt_type]
            for resume in resumes:
                prompt = template.format(resume=resume["text"])
                record = create_record(
                    client,
                    resume=resume,
                    condition=f"{prompt_type}_{version}",
                    prompt=prompt,
                    json_mode=False,
                )
                record["prompt_type"] = prompt_type
                record["version"] = version
                records.append(record)

    write_results(records)

    print(f"Model: {MODEL}")
    for condition in ("plain_prompting", "json_mode"):
        summary = summarize(
            [record for record in records if record["condition"] == condition]
        )
        print(
            f"{condition.replace('_', ' ').title()}: "
            f"{summary['invalid_json']}/{summary['total']} invalid JSON | "
            f"{summary['schema_valid']}/{summary['total']} schema-valid | "
            f"{summary['correct_extractions']}/{summary['total']} exact extractions"
        )

    print("Prompt rewrite results (invalid JSON: original -> rewritten):")
    for prompt_type in BAD_PROMPTS:
        summaries = {
            version: summarize(
                [
                    record
                    for record in records
                    if record["prompt_type"] == prompt_type
                    and record["version"] == version
                ]
            )
            for version in ("original", "rewritten")
        }
        original = summaries["original"]
        rewritten = summaries["rewritten"]
        print(
            f"  {prompt_type}: {original['invalid_json']}/{original['total']} -> "
            f"{rewritten['invalid_json']}/{rewritten['total']} invalid; "
            f"schema-valid {original['schema_valid']}/{original['total']} -> "
            f"{rewritten['schema_valid']}/{rewritten['total']}"
        )

    json_records = [
        record for record in records if record["condition"] == "json_mode"
    ]
    brier_score = confidence_brier_score(json_records)
    if brier_score is None:
        print("Confidence calibration: unavailable (no schema-valid JSON-mode outputs).")
    else:
        print(
            "Confidence Brier score against exact extraction correctness "
            f"(lower is better; n={sum(record['parsed'] is not None for record in json_records)}): "
            f"{brier_score:.3f}"
        )
    print("Saved raw outputs to experiment_results.json and prompt summaries to prompt_rewrite_results.csv.")


if __name__ == "__main__":
    main()
