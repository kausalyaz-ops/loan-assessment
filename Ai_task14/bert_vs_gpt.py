"""Compare BERT fill-mask predictions with GPT-2 text generation."""

from pathlib import Path

from torch.nn import Module
from transformers import pipeline


FILL_MASK_SENTENCES = [
    "The capital of France is [MASK].",
    "A dog is a type of [MASK].",
    "She wrote a letter with a [MASK].",
]

GENERATION_PROMPTS = [
    "The benefits of regular exercise include",
    "A good customer support reply should",
    "Artificial intelligence is changing education because",
]

COMPARISON_TABLE = """| Aspect | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only Transformer | Decoder-only Transformer |
| Reads / writes | Reads both left and right context; fills a masked token | Reads preceding context; generates text from left to right |
| Best for | Understanding and classifying text | Generating coherent text continuations |
| Example tasks | Ticket classification, sentiment analysis, masked-word prediction | Drafting replies, creative writing, text completion |
"""

MODEL_CHOICE = """BERT is well suited to ticket classification because its encoder uses context from both sides of each word.
It produces contextual representations that help distinguish ticket intent and category.
GPT is well suited to drafting replies because its decoder generates fluent text one token at a time.
Given a ticket and instructions, GPT can continue the prompt with a relevant, natural-language response.
"""


def count_parameters(model: Module) -> int:
    """Return the number of parameters in a Hugging Face pipeline's model."""
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    text_generation = pipeline("text-generation", model="gpt2")
    text_generation.tokenizer.pad_token = text_generation.tokenizer.eos_token

    bert_parameter_count = count_parameters(fill_mask.model)
    gpt_parameter_count = count_parameters(text_generation.model)

    report = [
        "# BERT vs GPT",
        "",
        "## BERT fill-mask predictions",
        "",
    ]
    for sentence in FILL_MASK_SENTENCES:
        prediction = fill_mask(sentence, top_k=1)[0]
        report.append(f"- **{sentence}** -> `{prediction['token_str'].strip()}`")

    report.extend(["", "## GPT-2 generated continuations", ""])
    for prompt in GENERATION_PROMPTS:
        result = text_generation(
            prompt,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=30,
            pad_token_id=text_generation.tokenizer.eos_token_id,
        )[0]["generated_text"]
        continuation = result[len(prompt) :].strip()
        report.append(f"- **Prompt:** {prompt}\n  **Continuation:** {continuation}")

    report.extend(
        [
            "",
            "## Parameter counts",
            "",
            f"- `bert-base-uncased`: {bert_parameter_count:,} parameters",
            f"- `gpt2`: {gpt_parameter_count:,} parameters",
            "",
            "## Comparison",
            "",
            COMPARISON_TABLE.rstrip(),
            "",
            "## Why choose each model?",
            "",
            MODEL_CHOICE.rstrip(),
            "",
        ]
    )

    report_path = Path(__file__).with_name("submission.md")
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved submission report to: {report_path}")


if __name__ == "__main__":
    main()
