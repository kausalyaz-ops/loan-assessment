"""Train and compare a tiny LSTM and a two-layer Transformer encoder."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


SEED = 42
SEQUENCE_LENGTH = 12
TRAIN_SAMPLES = 1024
TEST_SAMPLES = 256
EPOCHS = 100
BATCH_SIZE = 64
EMBEDDING_DIMENSION = 32
LEARNING_RATE = 0.003

MANAGER_EXPLANATION = """ChatGPT generates a response by repeatedly predicting what should come next. First, it breaks your prompt into pieces called tokens, which may be words or parts of words. It turns those tokens into numbers and uses the surrounding context to estimate a probability for many possible next tokens. For example, after 'The meeting starts at,' it may consider times or other likely continuations. The system selects one candidate, then adds it to the text already written. It uses the expanded text to make another prediction, repeating this process until it reaches a stopping point or a length limit. The model does not plan every sentence in advance; each choice influences later choices. Instructions and conversation history help shape those probabilities, so the same model can answer questions, draft messages, or summarize documents. Its fluency does not guarantee correctness: it can produce confident mistakes, so important claims still need human review."""

TRANSFORMER_REASONS = [
    "Parallelism: unlike a recurrent network, a Transformer can process all input positions together during training, making better use of modern accelerators.",
    "Long-range attention: self-attention can directly connect distant tokens instead of passing information step by step through a recurrent state.",
    "Scaling: the architecture scales effectively with larger datasets, model sizes, and compute, enabling strong general-purpose capabilities.",
    "Transfer learning: pretrained Transformer models can be adapted to many downstream tasks, reducing the need to train each model from scratch.",
]

VANISHING_GRADIENT_EXPLANATION = """During backpropagation, gradients are multiplied across many time steps.
If those factors are repeatedly smaller than one, the gradient shrinks toward zero.
Early sequence steps then receive almost no learning signal.
The network struggles to learn relationships between events far apart in the sequence."""

KV_CACHE_EXPLANATION = """During autoregressive generation, the model computes key and value vectors for each new token in every attention layer. KV caching stores those vectors so they do not need to be recomputed at every next-token step. The model processes only the newest token against the cached history, reducing repeated computation and speeding up long responses. The cache uses additional memory, which grows with sequence length."""


class LSTMClassifier(nn.Module):
    """Classify whether a digit sequence is nondecreasing."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, EMBEDDING_DIMENSION)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIMENSION,
            hidden_size=EMBEDDING_DIMENSION,
            batch_first=True,
        )
        self.classifier = nn.Linear(EMBEDDING_DIMENSION, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden[-1])


class TransformerClassifier(nn.Module):
    """Classify digit sequences with a two-layer Transformer encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, EMBEDDING_DIMENSION)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH + 1, EMBEDDING_DIMENSION)
        self.class_token = nn.Parameter(torch.zeros(1, 1, EMBEDDING_DIMENSION))
        nn.init.normal_(self.class_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIMENSION,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(EMBEDDING_DIMENSION, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embedding(tokens)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        embedded = torch.cat((class_token, token_embeddings), dim=1)
        positions = torch.arange(embedded.shape[1], device=tokens.device)
        encoded = self.encoder(embedded + self.position_embedding(positions))
        return self.classifier(encoded[:, 0])


def create_dataset(sample_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create balanced sorted and unsorted sequences with matching digit counts."""
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(sample_count // 2):
        while True:
            sorted_sequence = np.sort(rng.integers(0, 10, size=SEQUENCE_LENGTH))
            unequal_pairs = np.flatnonzero(sorted_sequence[:-1] != sorted_sequence[1:])
            if unequal_pairs.size:
                break

        unsorted_sequence = sorted_sequence.copy()
        swap_position = int(rng.choice(unequal_pairs))
        unsorted_sequence[swap_position], unsorted_sequence[swap_position + 1] = (
            unsorted_sequence[swap_position + 1],
            unsorted_sequence[swap_position],
        )
        sequences.extend((sorted_sequence, unsorted_sequence))
        labels.extend((1, 0))

    order = rng.permutation(len(labels))
    tokens = torch.tensor(np.asarray(sequences)[order].copy(), dtype=torch.long)
    targets = torch.tensor(np.asarray(labels)[order].copy(), dtype=torch.long)
    return tokens, targets


def train_and_measure(
    model: nn.Module,
    train_tokens: torch.Tensor,
    train_targets: torch.Tensor,
    test_tokens: torch.Tensor,
    test_targets: torch.Tensor,
) -> tuple[float, float]:
    """Train for a fixed number of epochs and return test accuracy and train time."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)

    start_time = time.perf_counter()
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(train_tokens.shape[0], generator=generator)
        for indices in order.split(BATCH_SIZE):
            logits = model(train_tokens[indices])
            loss = loss_function(logits, train_targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    training_seconds = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        predictions = model(test_tokens).argmax(dim=1)
        accuracy = (predictions == test_targets).float().mean().item()
    return accuracy, training_seconds


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 2))

    train_tokens, train_targets = create_dataset(TRAIN_SAMPLES, SEED)
    test_tokens, test_targets = create_dataset(TEST_SAMPLES, SEED + 1)
    model_results: list[tuple[str, float, float]] = []

    for name, model_type in (
        ("LSTM", LSTMClassifier),
        ("TransformerEncoder (2 layers)", TransformerClassifier),
    ):
        torch.manual_seed(SEED)
        model = model_type()
        accuracy, training_seconds = train_and_measure(
            model, train_tokens, train_targets, test_tokens, test_targets
        )
        model_results.append((name, accuracy, training_seconds))

    report_path = Path(__file__).resolve().with_name("submission.md")
    result_rows = "\n".join(
        f"| {name} | {accuracy:.1%} | {training_seconds:.2f} |"
        for name, accuracy, training_seconds in model_results
    )
    reasons = "\n".join(f"{index}. {reason}" for index, reason in enumerate(TRANSFORMER_REASONS, 1))
    explanation_word_count = len(MANAGER_EXPLANATION.split())
    if explanation_word_count != 150:
        raise ValueError(
            f"Manager explanation must be exactly 150 words, got {explanation_word_count}."
        )

    report = f"""# Transformers vs RNNs

## Toy sequence task and results

Both models were trained to classify length-{SEQUENCE_LENGTH} digit sequences as nondecreasing (sorted) or not sorted. The balanced dataset pairs each sorted sequence with an unsorted version that has the same digits, split into {TRAIN_SAMPLES} training and {TEST_SAMPLES} held-out examples. Each model was trained for {EPOCHS} epochs with the same batch size and optimizer.

| Model | Held-out accuracy | Training time (seconds) |
|---|---:|---:|
{result_rows}

Training time is measured on the machine running this script and includes training epochs only, not model initialization or test evaluation. Results are for this small synthetic task; they do not represent general benchmark rankings.

## Four reasons Transformers replaced RNNs

{reasons}

## Vanishing gradients in four lines

{VANISHING_GRADIENT_EXPLANATION}

## How ChatGPT-style models generate text

{MANAGER_EXPLANATION}

*Word count: {explanation_word_count}.*

## Bonus: KV caching

{KV_CACHE_EXPLANATION}
"""
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved submission to: {report_path}")


if __name__ == "__main__":
    main()
