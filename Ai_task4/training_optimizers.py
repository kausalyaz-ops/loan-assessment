from copy import deepcopy
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.05


@dataclass
class TrainingResult:
    train_losses: list[float]
    validation_losses: list[float]
    learning_rates: list[float]
    best_validation_loss: float
    best_epoch: int
    epochs_run: int


class MoonClassifier(nn.Module):
    def __init__(self, use_regularization: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, 32)]
        if use_regularization:
            layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        if use_regularization:
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(32, 32))
        if use_regularization:
            layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        if use_regularization:
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(32, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def make_loaders() -> tuple[DataLoader, DataLoader]:
    features, labels = make_moons(n_samples=1000, noise=0.2, random_state=SEED)
    train_x, validation_x, train_y, validation_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32).unsqueeze(1),
    )
    validation_dataset = TensorDataset(
        torch.tensor(validation_x, dtype=torch.float32),
        torch.tensor(validation_y, dtype=torch.float32).unsqueeze(1),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, validation_loader


def make_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
) -> torch.optim.Optimizer:
    if name == "SGD":
        return torch.optim.SGD(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if name == "SGD with momentum":
        return torch.optim.SGD(
            parameters,
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer: {name}")


def evaluate(
    model: MoonClassifier,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for features, labels in loader:
            batch_loss = criterion(model(features), labels)
            total_loss += batch_loss.item() * len(features)
            total_examples += len(features)
    return total_loss / total_examples


def train_model(
    optimizer_name: str,
    use_regularization: bool,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> TrainingResult:
    torch.manual_seed(SEED)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(SEED)

    model = MoonClassifier(use_regularization)
    optimizer = make_optimizer(optimizer_name, model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    train_losses: list[float] = []
    validation_losses: list[float] = []
    learning_rates: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            batch_loss = criterion(model(features), labels)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * len(features)
            total_examples += len(features)

        train_losses.append(total_loss / total_examples)
        validation_loss = evaluate(model, validation_loader, criterion)
        validation_losses.append(validation_loss)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step()
        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return TrainingResult(
        train_losses=train_losses,
        validation_losses=validation_losses,
        learning_rates=learning_rates,
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        epochs_run=len(train_losses),
    )


def plot_loss_curves(results: dict[str, TrainingResult]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, result in results.items():
        losses = result.train_losses
        ax.plot(range(1, len(losses) + 1), losses, label=name)

    ax.set(
        title="make_moons training loss by optimizer",
        xlabel="Epoch",
        ylabel="Binary cross-entropy loss",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("optimizer_loss_curves.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_validation_comparison(
    results: dict[tuple[str, bool], TrainingResult],
) -> Path:
    names = ("SGD", "SGD with momentum", "Adam")
    baseline_losses = [
        results[(name, False)].best_validation_loss for name in names
    ]
    regularized_losses = [
        results[(name, True)].best_validation_loss for name in names
    ]
    positions = range(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [position - width / 2 for position in positions],
        baseline_losses,
        width,
        label="Without Dropout/BatchNorm",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        regularized_losses,
        width,
        label="With Dropout(0.3)/BatchNorm1d",
    )
    ax.set(
        title="Best validation loss: regularized vs. baseline",
        xlabel="Optimizer",
        ylabel="Binary cross-entropy loss",
        xticks=list(positions),
        xticklabels=names,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("validation_loss_comparison.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_learning_rates(
    results: dict[str, TrainingResult],
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, result in results.items():
        learning_rates = result.learning_rates
        ax.step(
            range(1, len(learning_rates) + 1),
            learning_rates,
            where="mid",
            label=name,
        )

    ax.set(
        title="StepLR learning rate over time",
        xlabel="Epoch",
        ylabel="Learning rate",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("step_lr_schedule.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_hyperparameter_table() -> Path:
    content = """# Hyperparameter guide

| Hyperparameter | If too low | If too high |
|---|---|---|
| Learning rate | Learning is very slow or stalls before a useful solution. | Updates overshoot; loss may oscillate or diverge. |
| Batch size | Noisy updates and slower training from frequent updates. | Smoother updates but more memory, fewer updates per epoch, and potentially poorer generalization. |
| Epochs | The model may underfit because training stops too soon. | Training takes longer and may overfit without early stopping or other regularization. |
| Weight decay | Little constraint on large weights; overfitting may increase. | Weights are shrunk too strongly; the model may underfit. |

Adam adapts each parameter's update using moving averages of gradients and squared gradients.
Adam with weight decay applies the decay through Adam's gradient update, which couples decay and adaptation.
AdamW decouples weight decay from the gradient update, making regularization more direct and predictable.
"""
    output_path = Path(__file__).with_name("hyperparameter_guide.md")
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    torch.manual_seed(SEED)
    train_loader, validation_loader = make_loaders()
    optimizers = ("SGD", "SGD with momentum", "Adam")
    results: dict[tuple[str, bool], TrainingResult] = {}

    print(
        f"make_moons: 800 training examples, 200 validation examples; "
        f"batch size {BATCH_SIZE}, up to {EPOCHS} epochs."
    )
    print(f"Early stopping patience: {PATIENCE}; StepLR: step_size=25, gamma=0.5.")

    for optimizer_name in optimizers:
        for use_regularization in (False, True):
            result = train_model(
                optimizer_name,
                use_regularization,
                train_loader,
                validation_loader,
            )
            results[(optimizer_name, use_regularization)] = result
            setup = "with Dropout/BatchNorm" if use_regularization else "baseline"
            print(
                f"{optimizer_name} ({setup}): "
                f"best validation loss = {result.best_validation_loss:.4f} "
                f"at epoch {result.best_epoch} "
                f"(stopped after {result.epochs_run} epochs)"
            )

    baseline_results = {
        name: results[(name, False)]
        for name in optimizers
    }
    loss_chart = plot_loss_curves(baseline_results)
    validation_chart = plot_validation_comparison(results)
    scheduler_chart = plot_learning_rates(baseline_results)
    table_path = write_hyperparameter_table()

    print("\nSubmission files:")
    for path in (loss_chart, validation_chart, scheduler_chart, table_path):
        print(path)
    print(
        "\nConcepts: SGD, Adam, AdamW, Dropout, BatchNorm, "
        "early stopping, StepLR"
    )


if __name__ == "__main__":
    main()
