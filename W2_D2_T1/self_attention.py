"""Compute scaled dot-product self-attention for four three-dimensional tokens."""

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    """Apply numerically stable softmax over the last axis."""
    shifted_values = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_values)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def main() -> None:
    np.random.seed(0)

    tokens = np.random.rand(4, 3)
    w_q = np.random.rand(3, 3)
    w_k = np.random.rand(3, 3)
    w_v = np.random.rand(3, 3)

    queries = tokens @ w_q
    keys = tokens @ w_k
    values = tokens @ w_v

    scores = queries @ keys.T / np.sqrt(queries.shape[1])
    attention_weights = softmax(scores)
    output = attention_weights @ values

    print("Attention weights (4x4):")
    print(attention_weights)
    print("\nRow sums:")
    print(attention_weights.sum(axis=1))
    print("\nAttention output:")
    print(output)

    np.testing.assert_allclose(attention_weights.sum(axis=1), np.ones(4))


if __name__ == "__main__":
    main()
