"""Demonstrate two-head attention and inspect a pretrained BERT attention map."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


SENTENCE = "The bank by the river was steep"
MODEL_NAME = "bert-base-uncased"
HEAD_COUNT = 2
MODEL_DIMENSION = 4
HEAD_DIMENSION = MODEL_DIMENSION // HEAD_COUNT


def softmax(scores: np.ndarray) -> np.ndarray:
    """Apply numerically stable softmax over the last dimension."""
    shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_scores)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def run_two_head_attention() -> np.ndarray:
    """Split projected 4D token vectors across two heads and combine them."""
    np.random.seed(0)
    token_vectors = np.random.rand(5, MODEL_DIMENSION)
    w_q = np.random.rand(MODEL_DIMENSION, MODEL_DIMENSION)
    w_k = np.random.rand(MODEL_DIMENSION, MODEL_DIMENSION)
    w_v = np.random.rand(MODEL_DIMENSION, MODEL_DIMENSION)
    w_o = np.random.rand(MODEL_DIMENSION, MODEL_DIMENSION)

    queries = (token_vectors @ w_q).reshape(-1, HEAD_COUNT, HEAD_DIMENSION)
    keys = (token_vectors @ w_k).reshape(-1, HEAD_COUNT, HEAD_DIMENSION)
    values = (token_vectors @ w_v).reshape(-1, HEAD_COUNT, HEAD_DIMENSION)

    queries = queries.transpose(1, 0, 2)
    keys = keys.transpose(1, 0, 2)
    values = values.transpose(1, 0, 2)

    scores = queries @ keys.transpose(0, 2, 1) / np.sqrt(HEAD_DIMENSION)
    head_weights = softmax(scores)
    head_outputs = head_weights @ values

    concatenated_heads = head_outputs.transpose(1, 0, 2).reshape(
        -1, MODEL_DIMENSION
    )
    return concatenated_heads @ w_o


def inspect_bert_attention() -> None:
    """Print the bank-token attention row and save one head as a heatmap."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    inputs = tokenizer(SENTENCE, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    bank_positions = [
        index for index, token in enumerate(tokens) if token.lower() == "bank"
    ]
    if len(bank_positions) != 1:
        raise ValueError(
            f"Expected one 'bank' token in tokenizer output, got {bank_positions}: "
            f"{tokens}"
        )
    bank_position = bank_positions[0]

    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True)

    if outputs.attentions is None:
        raise RuntimeError("The model did not return attention weights.")

    selected_head = 0
    attention_matrix = outputs.attentions[-1][0, selected_head].cpu().numpy()
    print(f"\nBERT tokens: {tokens}")
    print(f"Last-layer head {selected_head} attention for 'bank':")
    for token, weight in zip(tokens, attention_matrix[bank_position]):
        print(f"  {token:>12}: {weight:.6f}")

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(attention_matrix, cmap="viridis", aspect="auto")
    axis.set(
        xticks=np.arange(len(tokens)),
        yticks=np.arange(len(tokens)),
        xticklabels=tokens,
        yticklabels=tokens,
        xlabel="Key token",
        ylabel="Query token",
        title=f"BERT last-layer head {selected_head} attention",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.colorbar(image, ax=axis, label="Attention weight")
    figure.tight_layout()

    heatmap_path = Path(__file__).with_name("attention_heatmap.png")
    figure.savefig(heatmap_path, dpi=150)
    plt.close(figure)
    print(f"\nSaved token-labeled attention heatmap to: {heatmap_path}")


def main() -> None:
    combined_output = run_two_head_attention()
    print("Two-head output after concatenation and output projection:")
    print(combined_output)
    inspect_bert_attention()


if __name__ == "__main__":
    main()
