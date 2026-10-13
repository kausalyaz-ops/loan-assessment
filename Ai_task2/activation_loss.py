from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean((y_true - y_pred) ** 2))


def cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_prob must be matching 2D arrays")
    clipped_prob = np.clip(y_prob, 1e-12, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(clipped_prob), axis=1)))


def plot_activations() -> Path:
    x = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.maximum(0, x), label="ReLU")
    ax.plot(x, 1 / (1 + np.exp(-x)), label="Sigmoid")
    ax.plot(x, np.tanh(x), label="Tanh")
    ax.set(title="Activation functions", xlabel="x", ylabel="f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("activation_functions.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = softmax(logits)
    assert np.isclose(probabilities.sum(), 1.0)
    print(
        f"softmax({logits.tolist()}) = {np.round(probabilities, 3)} "
        f"(sum {probabilities.sum():.1f})"
    )

    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.9, 0.2, 0.8])
    print(f"MSE on 3 predictions: {mse(y_true, y_pred):.3f}")

    class_labels = np.eye(3)
    class_probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    print(
        "Cross-entropy on 3 predictions: "
        f"{cross_entropy(class_labels, class_probabilities):.3f}"
    )

    true_class = np.array([[1.0, 0.0]])
    confident_wrong = cross_entropy(true_class, np.array([[0.01, 0.99]]))
    uncertain = cross_entropy(true_class, np.array([[0.4, 0.6]]))
    assert confident_wrong > uncertain
    print(f"Confident wrong cross-entropy: {confident_wrong:.3f}")
    print(f"Uncertain cross-entropy: {uncertain:.3f}")

    chart_path = plot_activations()
    print(f"Activation plot saved to: {chart_path}")

    print("House price: linear output activation; mean squared error loss.")
    print("Spam yes/no: sigmoid output activation; binary cross-entropy loss.")
    print("5-class image label: softmax output activation; categorical cross-entropy loss.")
    print("Hidden layers: ReLU is a common activation for all three tasks.")


if __name__ == "__main__":
    main()
