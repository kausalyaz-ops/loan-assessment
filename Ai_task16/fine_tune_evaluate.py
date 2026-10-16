"""Fine-tune and evaluate DistilBERT for emotion classification."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.trainer_callback import TrainerCallback


MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_SAMPLES = 2_000
MAX_LENGTH = 128
SEED = 42
EPOCHS = 3
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MODEL_DIR = OUTPUT_DIR / "distilbert-emotion"
FROZEN_MODEL_DIR = OUTPUT_DIR / "distilbert-emotion-frozen"

NEW_SENTENCES = [
    "I finally got the job I wanted and could not be happier.",
    "I feel nervous about the medical test tomorrow.",
    "My friend surprised me with a thoughtful birthday gift.",
    "I am so frustrated that the train was cancelled again.",
    "I miss spending time with my family.",
]


class EpochMetricsCallback(TrainerCallback):
    """Print validation accuracy and F1 as each epoch finishes."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        accuracy = metrics.get("eval_accuracy")
        f1 = metrics.get("eval_f1")
        if accuracy is not None and f1 is not None:
            epoch = int(round(state.epoch or 0))
            print(f"Epoch {epoch}: accuracy {accuracy:.2f}, F1 {f1:.2f}")
        return control


def load_splits():
    dataset = load_dataset("dair-ai/emotion", "split", split="train")
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    label_names = dataset.features["label"].names
    splits = dataset.train_test_split(
        test_size=0.2,
        seed=SEED,
        stratify_by_column="label",
    )
    splits["validation"] = splits.pop("test")
    return splits, label_names


def tokenize_splits(splits, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = splits.map(tokenize, batched=True)
    return tokenized.rename_column("label", "labels")


def create_model(label_names):
    label2id = {label: index for index, label in enumerate(label_names)}
    id2label = {index: label for label, index in label2id.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def create_trainer(model, tokenizer, tokenized_splits, output_dir):
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        weight_decay=0.01,
        report_to="none",
        seed=SEED,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EpochMetricsCallback()],
    )


def evaluate_and_print_confusion_matrix(trainer, tokenized_validation, label_names):
    results = trainer.predict(tokenized_validation)
    predictions = np.argmax(results.predictions, axis=-1)
    print(
        "Final validation metrics: "
        f"accuracy {results.metrics['test_accuracy']:.2f}, "
        f"F1 {results.metrics['test_f1']:.2f}"
    )
    print("Validation confusion matrix (rows=true, columns=predicted):")
    print(f"Labels: {label_names}")
    print(confusion_matrix(results.label_ids, predictions, labels=range(len(label_names))))
    return results.metrics


def train_frozen_head(tokenizer, tokenized_splits, label_names):
    model = create_model(label_names)
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False

    trainer = create_trainer(model, tokenizer, tokenized_splits, FROZEN_MODEL_DIR)
    start_time = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - start_time
    metrics = trainer.evaluate()
    return metrics, training_seconds


def main():
    print(f"CUDA GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits, label_names = load_splits()
    print(
        f"Training examples: {len(splits['train']):,}; "
        f"validation examples: {len(splits['validation']):,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_splits = tokenize_splits(splits, tokenizer)

    baseline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=20_000),
        LogisticRegression(max_iter=1_000, random_state=SEED),
    )
    baseline_start = perf_counter()
    baseline.fit(splits["train"]["text"], splits["train"]["label"])
    baseline_seconds = perf_counter() - baseline_start
    baseline_predictions = baseline.predict(splits["validation"]["text"])
    baseline_accuracy = accuracy_score(splits["validation"]["label"], baseline_predictions)
    baseline_f1 = f1_score(
        splits["validation"]["label"],
        baseline_predictions,
        average="weighted",
    )
    print(
        f"TF-IDF + Logistic Regression baseline: "
        f"accuracy {baseline_accuracy:.2f}, F1 {baseline_f1:.2f} "
        f"({baseline_seconds:.1f}s)"
    )

    trainer = create_trainer(
        create_model(label_names),
        tokenizer,
        tokenized_splits,
        MODEL_DIR,
    )
    training_start = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - training_start

    validation_metrics = evaluate_and_print_confusion_matrix(
        trainer,
        tokenized_splits["validation"],
        label_names,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Saved model and tokenizer to {MODEL_DIR}")

    classifier = pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        device=0 if torch.cuda.is_available() else -1,
    )
    print("\nPredictions for new sentences:")
    for sentence in NEW_SENTENCES:
        prediction = classifier(sentence)[0]
        print(
            f"- {sentence}\n"
            f"  label: {prediction['label']}, confidence: {prediction['score']:.4f}"
        )

    print(
        f"\nFull fine-tuning: accuracy {validation_metrics['test_accuracy']:.2f}, "
        f"F1 {validation_metrics['test_f1']:.2f}, training time {training_seconds:.1f}s"
    )

    frozen_metrics, frozen_seconds = train_frozen_head(
        tokenizer,
        tokenized_splits,
        label_names,
    )
    print(
        "Frozen feature extractor (classifier head only): "
        f"accuracy {frozen_metrics['eval_accuracy']:.2f}, "
        f"F1 {frozen_metrics['eval_f1']:.2f}, training time {frozen_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
