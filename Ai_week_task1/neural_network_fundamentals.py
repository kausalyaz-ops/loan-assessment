from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    clipped_values = np.clip(values, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


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


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    clipped_prob = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(
            y_true * np.log(clipped_prob)
            + (1.0 - y_true) * np.log(1.0 - clipped_prob)
        )
    )


@dataclass
class NetworkParameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclass
class TrainingResult:
    parameters: NetworkParameters
    losses: list[float]
    diverged_epoch: int | None


def initialize_parameters(seed: int) -> NetworkParameters:
    rng = np.random.default_rng(seed)
    return NetworkParameters(
        w1=rng.normal(0.0, np.sqrt(2.0 / 2.0), size=(2, 4)),
        b1=np.zeros((1, 4)),
        w2=rng.normal(0.0, np.sqrt(2.0 / 4.0), size=(4, 1)),
        b2=np.zeros((1, 1)),
    )


def forward(
    inputs: np.ndarray, parameters: NetworkParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_linear = inputs @ parameters.w1 + parameters.b1
    hidden = relu(hidden_linear)
    output = sigmoid(hidden @ parameters.w2 + parameters.b2)
    return hidden_linear, hidden, output


def train_network(
    learning_rate: float, epochs: int = 5000, seed: int = 39
) -> TrainingResult:
    parameters = initialize_parameters(seed)
    losses: list[float] = []
    diverged_epoch: int | None = None
    sample_count = X.shape[0]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for epoch in range(epochs):
            hidden_linear, hidden, predictions = forward(X, parameters)
            loss_value = binary_cross_entropy(Y, predictions)
            if not np.isfinite(loss_value):
                diverged_epoch = epoch + 1
                losses.append(float("inf"))
                break
            losses.append(loss_value)

            output_gradient = (predictions - Y) / sample_count
            w2_gradient = hidden.T @ output_gradient
            b2_gradient = np.sum(output_gradient, axis=0, keepdims=True)
            hidden_gradient = output_gradient @ parameters.w2.T
            hidden_linear_gradient = hidden_gradient * (hidden_linear > 0.0)
            w1_gradient = X.T @ hidden_linear_gradient
            b1_gradient = np.sum(hidden_linear_gradient, axis=0, keepdims=True)

            gradients = (
                w1_gradient,
                b1_gradient,
                w2_gradient,
                b2_gradient,
            )
            if not all(np.all(np.isfinite(gradient)) for gradient in gradients):
                diverged_epoch = epoch + 1
                losses.append(float("inf"))
                break

            parameters.w1 -= learning_rate * w1_gradient
            parameters.b1 -= learning_rate * b1_gradient
            parameters.w2 -= learning_rate * w2_gradient
            parameters.b2 -= learning_rate * b2_gradient

            if not all(
                np.all(np.isfinite(value))
                for value in (
                    parameters.w1,
                    parameters.b1,
                    parameters.w2,
                    parameters.b2,
                )
            ):
                diverged_epoch = epoch + 1
                losses.append(float("inf"))
                break

    if diverged_epoch is None:
        _, _, predictions = forward(X, parameters)
        losses.append(binary_cross_entropy(Y, predictions))

    return TrainingResult(parameters, losses, diverged_epoch)


def plot_loss_curves(results: dict[float, TrainingResult]) -> Path:
    fig, (ax, zoom_ax) = plt.subplots(1, 2, figsize=(13, 5))
    for learning_rate, result in results.items():
        finite_losses = np.asarray(result.losses, dtype=float)
        finite_losses = finite_losses[np.isfinite(finite_losses)]
        epochs = np.arange(finite_losses.size)
        label = f"lr={learning_rate}"
        ax.plot(epochs, finite_losses, label=label)
        zoom_ax.plot(epochs, finite_losses, label=label)
        if result.diverged_epoch is not None:
            for plot_ax in (ax, zoom_ax):
                plot_ax.axvline(
                    result.diverged_epoch,
                    linestyle=":",
                    alpha=0.7,
                    label=f"lr={learning_rate} diverged",
                )

    ax.set_yscale("log")
    ax.set(
        title="XOR network training loss by learning rate",
        xlabel="Epoch",
        ylabel="Binary cross-entropy loss (log scale)",
    )
    zoom_ax.set_yscale("log")
    zoom_ax.set_xlim(0, 50)
    zoom_ax.set(
        title="First 50 epochs",
        xlabel="Epoch",
        ylabel="Binary cross-entropy loss (log scale)",
    )
    ax.grid(True, which="both", alpha=0.3)
    zoom_ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = Path(__file__).with_name("xor_loss_curves.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def verify_autograd() -> None:
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    loss = (w * 2.0 + b - 10.0) ** 2
    loss.backward()

    print("Hand gradient: dL/dw = 2*(2-10)*2 = -32; dL/db = 2*(2-10) = -16")
    print(f"Autograd: w.grad = {w.grad.item():.1f}, b.grad = {b.grad.item():.1f}")
    assert w.grad.item() == -32.0
    assert b.grad.item() == -16.0


def main() -> None:
    test_logits = np.array([2.0, 1.0, 0.1])
    print(f"softmax([2.0, 1.0, 0.1]) = {np.round(softmax(test_logits), 3)}")
    print(f"Example MSE: {mse(np.array([1.0, 0.0]), np.array([0.8, 0.2])):.3f}")
    print(
        "Example cross-entropy: "
        f"{cross_entropy(np.array([[1.0, 0.0]]), np.array([[0.8, 0.2]])):.3f}"
    )
    verify_autograd()

    results = {
        learning_rate: train_network(learning_rate)
        for learning_rate in (0.5, 0.01, 5.0)
    }
    standard_result = results[0.5]
    _, _, predictions = forward(X, standard_result.parameters)
    print(
        "\nXOR predictions after training (lr=0.5): "
        f"{np.round(predictions.ravel(), 2)}"
    )
    for learning_rate, result in results.items():
        final_loss = result.losses[-1]
        finite_losses = [loss for loss in result.losses if np.isfinite(loss)]
        peak_loss = max(finite_losses)
        if result.diverged_epoch is not None:
            print(
                f"lr={learning_rate}: diverged at epoch "
                f"{result.diverged_epoch} (loss became non-finite)"
            )
        elif peak_loss > 2.0 * finite_losses[0]:
            print(
                f"lr={learning_rate}: unstable loss spike "
                f"(peak = {peak_loss:.3f}); final loss = {final_loss:.6f}"
            )
        else:
            print(f"lr={learning_rate}: final loss = {final_loss:.6f}")

    plot_path = plot_loss_curves(results)
    print(f"Loss curves saved to: {plot_path}")

    print("\nRegularisation and training controls:")
    print("Dropout: Randomly disables a fraction of neurons during training.")
    print("It reduces co-adaptation and helps limit overfitting.")
    print("BatchNorm: Normalizes layer activations using batch statistics.")
    print("It stabilizes training and can allow faster optimization.")
    print("Early stopping: Stops training when validation loss stops improving.")
    print("It helps prevent overfitting and avoids unnecessary training.")
    print(
        "\nConcepts: forward pass, backpropagation, gradient descent, "
        "activations, loss functions, regularisation."
    )


if __name__ == "__main__":
    main()
