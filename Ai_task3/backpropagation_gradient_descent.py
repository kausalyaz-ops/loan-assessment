from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def loss(w: float, b: float, x: float, target: float) -> float:
    return (w * x + b - target) ** 2


def train(
    learning_rate: float,
    x: float = 2.0,
    target: float = 10.0,
    initial_w: float = 1.0,
    initial_b: float = 0.0,
    steps: int = 50,
) -> tuple[float, float, list[float]]:
    w = initial_w
    b = initial_b
    losses = [loss(w, b, x, target)]

    for _ in range(steps):
        error = w * x + b - target
        gradient_w = 2 * error * x
        gradient_b = 2 * error
        w -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
        losses.append(loss(w, b, x, target))

    return w, b, losses


def plot_loss_curves(curves: dict[float, list[float]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for learning_rate, losses in curves.items():
        ax.plot(range(len(losses)), losses, label=f"lr={learning_rate}")

    ax.set_yscale("log")
    ax.set(
        title="Gradient descent loss for different learning rates",
        xlabel="Step",
        ylabel="Loss (log scale)",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("loss_curves.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    x = 2.0
    target = 10.0
    initial_w = 1.0
    initial_b = 0.0
    initial_error = initial_w * x + initial_b - target
    gradient_w = 2 * initial_error * x
    gradient_b = 2 * initial_error
    print(f"Hand gradient at start: dL/dw = 2*(2-10)*2 = {gradient_w:.0f}, dL/db = {gradient_b:.0f}")

    learning_rates = (0.001, 0.05, 0.5)
    curves = {
        learning_rate: train(learning_rate, x=x, target=target)[2]
        for learning_rate in learning_rates
    }

    standard_losses = curves[0.05]
    print(f"\nGradient descent with lr=0.05 (initial loss: {standard_losses[0]:.1f})")
    for step in range(10, 51, 10):
        print(f"Step {step}: loss = {standard_losses[step]:.10f}")

    print("\nLearning-rate comparison after 50 steps:")
    for learning_rate, losses in curves.items():
        print(f"lr={learning_rate}: loss = {losses[-1]:.6g}")

    chart_path = plot_loss_curves(curves)
    print(f"\nLoss curves saved to: {chart_path}")

    print("\nGradient concepts:")
    print("1. A gradient is the derivative of the loss: it measures how the loss changes as a parameter changes.")
    print("2. The chain rule computes that derivative by multiplying derivatives through nested operations.")
    print("3. The gradient points toward the direction of steepest increase in loss.")
    print("4. We subtract a learning-rate-scaled gradient to move parameters toward lower loss.")
    print("\nBatch size is the number of training examples used for one parameter update.")
    print("An epoch is one complete pass through the full training dataset.")
    print("Concepts: chain rule, gradient descent, learning rate, epochs, batch size")


if __name__ == "__main__":
    main()
