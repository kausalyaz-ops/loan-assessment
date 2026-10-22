import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 8
RANDOM_SEED = 42
CSV_PATH = Path(__file__).resolve().parents[1] / "Ai_task9" / "breast_cancer.csv"
MODEL_PATH = Path(__file__).with_name("best_mlp_state_dict.pt")


class CsvDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        if feature_tensor.ndim != 2:
            raise ValueError("features must be a 2D array")
        if label_tensor.ndim != 1 or len(feature_tensor) != len(label_tensor):
            raise ValueError("labels must be 1D and match the number of feature rows")
        if len(label_tensor) == 0:
            raise ValueError("dataset must contain at least one row")
        self.features = feature_tensor
        self.labels = label_tensor

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class MLP(nn.Module):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 28),
            nn.BatchNorm1d(28),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(28, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def read_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[list[float]] = []
    labels: list[int] = []

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file is empty: {csv_path}")
        if len(header) < 2:
            raise ValueError("CSV must have at least one feature and one label column")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} columns; "
                    f"expected {len(header)}."
                )
            try:
                feature_rows.append([float(value) for value in row[:-1]])
                label_value = float(row[-1])
            except ValueError as error:
                raise ValueError(
                    f"CSV row {row_number} contains a non-numeric value."
                ) from error
            if not label_value.is_integer():
                raise ValueError(f"CSV row {row_number} has a non-integer class label.")
            labels.append(int(label_value))

    features = np.asarray(feature_rows, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("CSV features must be a non-empty matrix of finite numbers")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("This binary-classification example expects labels 0 and 1")
    return features, target


def validation_accuracy(model: MLP, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.size(0)
    if total == 0:
        raise ValueError("validation loader is empty")
    return correct / total


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    features, labels = read_csv(CSV_PATH)
    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(features[train_indices])
    scaled_validation_features = scaler.transform(features[validation_indices])
    train_dataset = CsvDataset(scaled_train_features, labels[train_indices])
    validation_dataset = CsvDataset(
        scaled_validation_features, labels[validation_indices]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MLP(input_features=features.shape[1])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Params: {parameter_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    training_losses: list[float] = []
    validation_accuracies: list[float] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        examples_seen = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            examples_seen += batch_size

        epoch_loss = total_loss / examples_seen
        epoch_accuracy = validation_accuracy(model, validation_loader)
        training_losses.append(epoch_loss)
        validation_accuracies.append(epoch_accuracy)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {epoch_loss:.4f} | val accuracy {epoch_accuracy:.4f}"
        )

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training completed without producing a model checkpoint")

    print(
        f"Best val accuracy {best_accuracy:.2f} at epoch {best_epoch} "
        f"({len(training_losses)} epochs trained)"
    )
    best_model = MLP(input_features=features.shape[1])
    best_model.load_state_dict(best_state_dict)
    best_model.eval()
    accuracy_before = validation_accuracy(best_model, validation_loader)

    torch.save(best_state_dict, MODEL_PATH)
    reloaded_model = MLP(input_features=features.shape[1])
    reloaded_model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    reloaded_model.eval()
    accuracy_after = validation_accuracy(reloaded_model, validation_loader)
    if accuracy_before != accuracy_after:
        raise RuntimeError("Validation accuracy changed after loading the state_dict")
    print(f"Reloaded model accuracy: {accuracy_after:.2f} (identical)")
    print(f"Best state_dict saved to: {MODEL_PATH}")

    logistic_regression = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    logistic_regression.fit(scaled_train_features, labels[train_indices])
    logistic_accuracy = logistic_regression.score(
        scaled_validation_features, labels[validation_indices]
    )
    print(f"Logistic regression validation accuracy: {logistic_accuracy:.2f}")

    print(
        "\nConcepts: tensors, nn.Module, Dataset, DataLoader, "
        "training loop, saving models."
    )


if __name__ == "__main__":
    main()
