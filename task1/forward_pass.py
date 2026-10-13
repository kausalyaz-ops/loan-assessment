import numpy as np


np.random.seed(1)

W1 = np.random.randn(3, 4)
b1 = np.zeros(4)
W2 = np.random.randn(4, 1)
b2 = np.zeros(1)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = relu(x @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return hidden, output


if __name__ == "__main__":
    x = np.array([0.5, -1.2, 3.0])
    hidden_values, output_probability = forward(x)

    print(f"W1: {W1.shape[0]}x{W1.shape[1]}")
    print(f"b1: {b1.size}")
    print(f"W2: {W2.shape[0]}x{W2.shape[1]}")
    print(f"b2: {b2.size}")
    print(f"Hidden values: {hidden_values}")
    print(f"Output probability: {output_probability[0]:.6f}")
