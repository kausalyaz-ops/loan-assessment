import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split


BATCH_SIZE = 32
CSV_PATH = Path(__file__).with_name("breast_cancer.csv")


class CsvDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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


def main() -> None:
    dataset = CsvDataset(CSV_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch_features, _ = next(iter(data_loader))

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(
        f"Train size: {len(train_dataset)} | "
        f"validation size: {len(validation_dataset)}"
    )
    print(
        "num_workers controls how many subprocesses load data: "
        "0 loads in the main process, while 2 can load batches in parallel "
        "but has process startup and communication overhead."
    )

    for worker_count in (0, 2):
        worker_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=worker_count,
        )
        worker_features, _ = next(iter(worker_loader))
        print(
            f"num_workers={worker_count} | "
            f"batch features: {worker_features.shape}"
        )

    batch_count = sum(1 for _ in data_loader)
    print(
        f"Dataset size: {len(dataset)} | "
        f"batch features: {batch_features.shape} | batches: {batch_count}"
    )


if __name__ == "__main__":
    main()
