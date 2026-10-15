import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split


BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
CSV_PATH = Path(__file__).resolve().parents[1] / "Ai_task9" / "breast_cancer.csv"
MODEL_PATH = Path(__file__).with_name("model.pt")
CHECKPOINT_PATH = Path(__file__).with_name("training_checkpoint.pt")
SCREENSHOT_PATH = Path(__file__).with_name("accuracy_comparison.png")


class BreastCancerDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, csv_path: str | Path) -> None:
        feature_rows: list[list[float]] = []
        labels: list[int] = []

        with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"CSV file is empty: {csv_path}")

            for row_number, row in enumerate(reader, start=2):
                if len(row) != 31:
                    raise ValueError(
                        f"Expected 30 features and 1 label on CSV row {row_number}; "
                        f"found {len(row)} columns."
                    )
                try:
                    feature_rows.append([float(value) for value in row[:-1]])
                    labels.append(int(row[-1]))
                except ValueError as error:
                    raise ValueError(
                        f"CSV row {row_number} contains a non-numeric value."
                    ) from error

        self.features = torch.tensor(feature_rows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class CancerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def train_epoch(
    model: CancerClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
) -> None:
    model.train()
    for features, labels in loader:
        optimizer.zero_grad()
        loss = loss_function(model(features), labels)
        loss.backward()
        optimizer.step()


def validation_accuracy(model: CancerClassifier, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.size(0)
    return correct / total


def main() -> None:
    torch.manual_seed(42)
    dataset = BreastCancerDataset(CSV_PATH)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_subset, validation_subset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_indices = train_subset.indices
    feature_mean = dataset.features[train_indices].mean(dim=0)
    feature_std = dataset.features[train_indices].std(dim=0).clamp_min(1e-8)
    normalized_dataset = torch.utils.data.TensorDataset(
        (dataset.features - feature_mean) / feature_std,
        dataset.labels,
    )
    train_dataset = Subset(normalized_dataset, train_subset.indices)
    validation_dataset = Subset(normalized_dataset, validation_subset.indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CancerClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, loss_function)

    accuracy_before = validation_accuracy(model, validation_loader)
    print(f"Validation accuracy before saving: {accuracy_before:.2%}")

    torch.save(model.state_dict(), MODEL_PATH)
    loaded_model = CancerClassifier()
    loaded_model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    loaded_model.eval()
    accuracy_after = validation_accuracy(loaded_model, validation_loader)
    print(f"Validation accuracy after loading:  {accuracy_after:.2%}")
    if accuracy_before != accuracy_after:
        raise RuntimeError("Validation accuracy changed after loading model weights.")

    def predict(features: torch.Tensor) -> tuple[int, float]:
        row = torch.as_tensor(features, dtype=torch.float32)
        if row.numel() != feature_mean.numel():
            raise ValueError(
                f"Expected one row with {feature_mean.numel()} features; "
                f"received {row.numel()} values."
            )
        normalized_row = (row.reshape(1, -1) - feature_mean) / feature_std
        with torch.no_grad():
            probabilities = loaded_model(normalized_row).softmax(dim=1)[0]
        label = int(probabilities.argmax().item())
        return label, float(probabilities[label].item())

    predicted_label, probability = predict(dataset.features[0])
    print(
        f"Example prediction: label={predicted_label}, "
        f"probability={probability:.2%}"
    )
    print(
        "Labels: 0 = malignant, 1 = benign. The probability is the predicted "
        "class's softmax probability."
    )
    print(
        "Saving state_dict stores only learned parameters, so the file is "
        "smaller and can be loaded into a model defined by the current code."
    )

    plt.figure(figsize=(7, 4))
    bars = plt.bar(
        ["Before saving", "After loading"],
        [accuracy_before * 100, accuracy_after * 100],
        color=["#4C78A8", "#59A14F"],
    )
    plt.ylim(0, 100)
    plt.ylabel("Validation accuracy (%)")
    plt.title("Model accuracy before saving and after loading")
    for bar, accuracy in zip(bars, (accuracy_before, accuracy_after)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 5,
            f"{accuracy:.2%}",
            ha="center",
            color="white",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(SCREENSHOT_PATH, dpi=160)
    plt.close()
    print(f"Accuracy comparison image saved to: {SCREENSHOT_PATH}")

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        CHECKPOINT_PATH,
    )
    resumed_model = CancerClassifier()
    resumed_optimizer = torch.optim.Adam(
        resumed_model.parameters(),
        lr=LEARNING_RATE,
    )
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    resumed_model.load_state_dict(checkpoint["model_state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_epoch = int(checkpoint["epoch"])
    train_epoch(resumed_model, train_loader, resumed_optimizer, loss_function)
    print(f"Resumed checkpoint at epoch {resumed_epoch}; trained one more epoch.")


if __name__ == "__main__":
    main()
