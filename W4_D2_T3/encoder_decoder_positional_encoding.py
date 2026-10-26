"""Compare encoder-only and decoder-only Transformers and plot positional encodings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline


POSITION_COUNT = 10
EMBEDDING_DIMENSION = 8
FILL_MASK_PROMPT = "The capital of France is [MASK]."
TEXT_GENERATION_PROMPT = "Once upon a time"

COMPARISON_TABLE = """| Aspect | Encoder-only (BERT-style) | Decoder-only (GPT-style) |
|---|---|---|
| How it reads | Uses self-attention over both left and right context in the input | Uses causal self-attention over preceding tokens only |
| What it outputs | Contextual representations; for fill-mask, scores candidate tokens at the masked position | A probability distribution for the next token, repeated to generate a continuation |
| Best tasks | Understanding: classification, sentiment analysis, extractive question answering, and fill-mask | Generation: text completion, drafting, summarization, and conversational responses |
| Example models | BERT, RoBERTa | GPT-2, GPT-3 |
"""

WORD_ORDER_EXPLANATION = """Without positional encoding, self-attention has no signal for token positions.
It treats the same words in a sentence as an unordered set of representations.
So “dog bites person” and “person bites dog” can look identical to the model.
Positional information lets the model distinguish sequence order and meaning.
"""


def sinusoidal_encoding(position_count: int, dimension: int) -> np.ndarray:
    """Return the standard sinusoidal positional encoding matrix."""
    positions = np.arange(position_count)[:, np.newaxis]
    dimensions = np.arange(dimension)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / dimension)
    angles = positions * angle_rates

    encoding = np.empty((position_count, dimension), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def save_heatmap(encoding: np.ndarray, output_path: Path) -> None:
    """Save positional encoding values as a labeled heatmap."""
    figure, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(encoding, cmap="coolwarm", aspect="auto")
    axis.set_title("Sinusoidal Positional Encoding (10 positions × 8 dimensions)")
    axis.set_xlabel("Embedding dimension")
    axis.set_ylabel("Position")
    axis.set_xticks(np.arange(encoding.shape[1]))
    axis.set_yticks(np.arange(encoding.shape[0]))
    figure.colorbar(image, ax=axis, label="Encoding value")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    output_directory = Path(__file__).resolve().parent
    heatmap_path = output_directory / "positional_encoding_heatmap.png"
    encoding = sinusoidal_encoding(POSITION_COUNT, EMBEDDING_DIMENSION)
    save_heatmap(encoding, heatmap_path)

    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    fill_mask_result = fill_mask(FILL_MASK_PROMPT, top_k=1)[0]

    text_generation = pipeline("text-generation", model="gpt2")
    text_generation.tokenizer.pad_token = text_generation.tokenizer.eos_token
    generation_result = text_generation(
        TEXT_GENERATION_PROMPT,
        do_sample=False,
        max_new_tokens=24,
        pad_token_id=text_generation.tokenizer.eos_token_id,
        return_full_text=False,
    )[0]["generated_text"].strip()

    report = f"""# Encoder vs Decoder and Positional Encoding

## Encoder-only vs decoder-only

{COMPARISON_TABLE.rstrip()}

## Sinusoidal positional encoding

The matrix contains {POSITION_COUNT} positions and {EMBEDDING_DIMENSION} dimensions.
For position `pos` and pair index `k`, dimensions `2k` and `2k+1` use
`sin(pos / 10000^(2k/d))` and `cos(pos / 10000^(2k/d))`, respectively.

![Heatmap of sinusoidal positional encoding](positional_encoding_heatmap.png)

## Why word order needs positional encoding

{WORD_ORDER_EXPLANATION.rstrip()}

## Model examples

- **BERT fill-mask** (`bert-base-uncased`): `{FILL_MASK_PROMPT}` -> **{fill_mask_result["token_str"].strip()}**.
  This is useful when predicting a missing word from context on both sides.
- **GPT-2 text-generation** (`gpt2`): **Prompt:** `{TEXT_GENERATION_PROMPT}`
  **Continuation:** {generation_result}
  This demonstrates autoregressive text continuation, producing one token after another.
"""
    report_path = output_directory / "submission.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved heatmap to: {heatmap_path}")
    print(f"Saved submission to: {report_path}")


if __name__ == "__main__":
    main()
