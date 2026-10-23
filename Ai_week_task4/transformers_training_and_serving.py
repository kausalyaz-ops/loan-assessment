import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import scan_cache_dir, snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)


RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-uncased"
SENTIMENT_PIPELINE_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
NER_PIPELINE_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"
NUM_EPOCHS = 3
MAX_LENGTH = 64
BATCH_SIZE = 16
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "fine_tuned_distilbert"
OUTPUT_DIR = BASE_DIR / "trainer_output"
COMPARISON_PATH = BASE_DIR / "comparison.csv"
SUBMISSION_PATH = BASE_DIR / "submission.md"
API_SCREENSHOT_PATH = BASE_DIR / "api_response.png"
LOCAL_FILES_ONLY = False
CLASS_NAMES = ("negative", "neutral", "positive")

TOKENIZER_EXAMPLES = (
    "The support team solved my problem quickly.",
    "I am disappointed with the delayed delivery.",
)
SENTIMENT_EXAMPLES = (
    "I love how easy this app is to use.",
    "The delivery arrived later than expected.",
    "The package contains the usual instruction booklet.",
)
NER_EXAMPLES = (
    "Sundar Pichai works at Google in California.",
    "Ada Lovelace was born in London.",
    "Microsoft opened an office in Paris.",
)


def create_labeled_dataset() -> tuple[list[str], np.ndarray]:
    """Return a deterministic, small three-class sentiment dataset."""
    phrases = {
        0: (
            "awful and disappointing",
            "terrible and frustrating",
            "poor and unpleasant",
            "broken and useless",
            "horrible and slow",
            "bad and unreliable",
            "annoying and confusing",
            "wasteful and disappointing",
            "damaged and disappointing",
            "unhelpful and stressful",
        ),
        1: (
            "average and ordinary",
            "okay and acceptable",
            "fine and typical",
            "standard and familiar",
            "moderate and reasonable",
            "regular and predictable",
            "neutral and unchanged",
            "adequate and routine",
            "neither good nor bad",
            "common and expected",
        ),
        2: (
            "excellent and delightful",
            "wonderful and satisfying",
            "great and reliable",
            "helpful and impressive",
            "pleasant and smooth",
            "fantastic and enjoyable",
            "loveable and convenient",
            "quick and dependable",
            "outstanding and comfortable",
            "useful and enjoyable",
        ),
    }
    products = (
        "phone",
        "movie",
        "meal",
        "book",
        "service",
        "app",
        "delivery",
        "camera",
        "hotel",
        "headphones",
    )
    templates = (
        "The {product} was {phrase}.",
        "I found the {product} {phrase}.",
        "My experience with the {product} felt {phrase}.",
        "Overall, this {product} seems {phrase}.",
        "Today the {product} proved {phrase}.",
    )

    texts: list[str] = []
    labels: list[int] = []
    samples_per_class = 60
    for label, class_phrases in phrases.items():
        for index in range(samples_per_class):
            phrase = class_phrases[index % len(class_phrases)]
            product = products[(index // len(class_phrases) + index) % len(products)]
            template = templates[index % len(templates)]
            texts.append(template.format(product=product, phrase=phrase))
            labels.append(label)
    return texts, np.asarray(labels, dtype=np.int64)


class EncodedTextDataset:
    def __init__(self, encodings: dict[str, list[list[int]]], labels: np.ndarray):
        import torch

        self.encodings = {
            key: [torch.tensor(row, dtype=torch.long) for row in value]
            for key, value in encodings.items()
        }
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: rows[index] for key, rows in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def compute_metrics(evaluation: Any) -> dict[str, float]:
    predictions = np.argmax(evaluation.predictions, axis=1)
    return {
        "accuracy": accuracy_score(evaluation.label_ids, predictions),
        "macro_f1": f1_score(
            evaluation.label_ids,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def show_tokenizer_examples(tokenizer: Any) -> None:
    encoded = tokenizer(
        list(TOKENIZER_EXAMPLES),
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )
    for index, sentence in enumerate(TOKENIZER_EXAMPLES):
        input_ids = encoded["input_ids"][index]
        attention_mask = encoded["attention_mask"][index]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(f"\nTokenizer sentence: {sentence}")
        print(f"Tokens:         {tokens}")
        print(f"Input ids:      {input_ids}")
        print(f"Attention mask: {attention_mask}")


def resolve_model_path(repository: str) -> str:
    for cached_repository in scan_cache_dir().repos:
        if cached_repository.repo_id == repository and cached_repository.revisions:
            revision = next(iter(cached_repository.revisions))
            snapshot_path = (
                Path(cached_repository.repo_path)
                / "snapshots"
                / revision.commit_hash
            )
            if snapshot_path.is_dir():
                return str(snapshot_path)
    if not LOCAL_FILES_ONLY:
        return snapshot_download(repository, local_files_only=False)
    raise FileNotFoundError(
        f"Model {repository!r} is not available in the local Hugging Face cache. "
        "Set LOCAL_FILES_ONLY = False to download it."
    )


def show_pipeline_examples() -> None:
    sentiment_model_path = resolve_model_path(SENTIMENT_PIPELINE_MODEL)
    ner_model_path = resolve_model_path(NER_PIPELINE_MODEL)
    sentiment = pipeline(
        "sentiment-analysis",
        model=sentiment_model_path,
        tokenizer=sentiment_model_path,
        device=-1,
    )
    ner = pipeline(
        "ner",
        model=ner_model_path,
        tokenizer=ner_model_path,
        aggregation_strategy="simple",
        device=-1,
    )
    print("\nSentiment-analysis pipeline:")
    for sentence in SENTIMENT_EXAMPLES:
        print(f"{sentence} -> {sentiment(sentence)[0]}")
    print("\nNER pipeline:")
    for sentence in NER_EXAMPLES:
        print(f"{sentence} -> {ner(sentence)}")


def time_inference(predictor: Any, texts: list[str], repeats: int = 3) -> float:
    samples = texts[: min(16, len(texts))]
    start = time.perf_counter()
    for _ in range(repeats):
        predictor(samples)
    elapsed = time.perf_counter() - start
    return elapsed / (repeats * len(samples))


def save_comparison_and_submission(
    results: list[dict[str, float | str]],
) -> None:
    import csv

    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "Model",
                "Accuracy",
                "Macro F1",
                "Training time (s)",
                "Inference time (ms/text)",
            ),
        )
        writer.writeheader()
        writer.writerows(results)

    comparison_table = [
        "| Model | Accuracy | Macro F1 | Training time (s) | Inference time (ms/text) |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        comparison_table.append(
            f"| {result['Model']} | {result['Accuracy']:.3f} "
            f"| {result['Macro F1']:.3f} | {result['Training time (s)']:.2f} "
            f"| {result['Inference time (ms/text)']:.2f} |"
        )

    report = "\n".join(
        [
            "# Transformers: Pipelines, Fine-tuning and Serving",
            "",
            "Dataset: generated short sentiment texts (60 examples per class). "
            "Metrics use a stratified 25% holdout.",
            "",
            "## TF-IDF vs fine-tuned DistilBERT",
            "",
            *comparison_table,
            "",
            "Training and inference times are measured on the machine running "
            "this script and are not directly comparable across hardware.",
            "",
            "## BERT encoder vs GPT decoder (5 lines)",
            "",
            "1. BERT is an encoder-only Transformer that reads context from both directions.",
            "2. GPT is a decoder-only Transformer that predicts the next token from left context.",
            "3. BERT is designed for understanding tasks such as classification and named-entity recognition.",
            "4. GPT is designed for text generation, continuation, and conversational responses.",
            "5. I would use DistilBERT for this sentiment task because it is compact and suited to classification.",
            "",
            "## API response screenshot",
            "",
            "See `api_response.png` for the captured successful POST `/predict` response.",
            "",
            "## Run the API",
            "",
            "Run `python transformers_training_and_serving.py` once to train and save the model, "
            "then start the service with `uvicorn api:app --host 127.0.0.1 --port 8000`.",
            "",
            "Example request: `curl -X POST http://127.0.0.1:8000/predict "
            '-H "Content-Type: application/json" -d "{\\"text\\":\\"I love this app\\"}"`.',
            "",
            "Whitespace-only and empty text are rejected with HTTP 422.",
            "",
        ]
    )
    SUBMISSION_PATH.write_text(report, encoding="utf-8")
    print(f"\nComparison table saved to: {COMPARISON_PATH}")
    print(f"Submission report saved to: {SUBMISSION_PATH}")


def save_api_screenshot() -> None:
    from fastapi.testclient import TestClient

    from api import app

    response = TestClient(app).post(
        "/predict",
        json={"text": "The support team was helpful and wonderful."},
    )
    response.raise_for_status()
    payload = response.json()

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    ax.text(
        0.03,
        0.85,
        "POST /predict  |  HTTP 200",
        fontsize=17,
        fontweight="bold",
        family="monospace",
    )
    ax.text(
        0.03,
        0.62,
        '{"text": "The support team was helpful and wonderful."}',
        fontsize=12,
        family="monospace",
    )
    ax.text(
        0.03,
        0.35,
        json.dumps(payload, indent=2),
        fontsize=13,
        family="monospace",
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f3f4f6", "edgecolor": "#9ca3af"},
    )
    fig.tight_layout()
    fig.savefig(API_SCREENSHOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"API response screenshot saved to: {API_SCREENSHOT_PATH}")

    invalid_response = TestClient(app).post("/predict", json={"text": "   "})
    if invalid_response.status_code != 422:
        raise RuntimeError(
            f"Expected whitespace-only text to return HTTP 422; "
            f"got {invalid_response.status_code}"
        )
    print("Empty-text validation: HTTP 422")


def main() -> None:
    import torch

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    base_model_path = resolve_model_path(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    show_tokenizer_examples(tokenizer)
    show_pipeline_examples()

    texts, labels = create_labeled_dataset()
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    tfidf_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            ),
        ]
    )
    start = time.perf_counter()
    tfidf_model.fit(train_texts, train_labels)
    tfidf_training_time = time.perf_counter() - start
    tfidf_predictions = tfidf_model.predict(validation_texts)
    tfidf_accuracy = accuracy_score(validation_labels, tfidf_predictions)
    tfidf_macro_f1 = f1_score(
        validation_labels,
        tfidf_predictions,
        average="macro",
        zero_division=0,
    )
    tfidf_inference_time = time_inference(tfidf_model.predict, validation_texts) * 1000

    encoded_train = tokenizer(
        train_texts,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded_validation = tokenizer(
        validation_texts,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    train_dataset = EncodedTextDataset(encoded_train, train_labels)
    validation_dataset = EncodedTextDataset(encoded_validation, validation_labels)
    label2id = {label: index for index, label in enumerate(CLASS_NAMES)}
    id2label = {index: label for label, index in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
    )
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=RANDOM_SEED,
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    start = time.perf_counter()
    trainer.train()
    transformer_training_time = time.perf_counter() - start
    evaluation = trainer.evaluate()
    transformer_accuracy = float(evaluation["eval_accuracy"])
    transformer_macro_f1 = float(evaluation["eval_macro_f1"])
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(MODEL_DIR)

    def transformer_predict(batch_texts: list[str]) -> list[int]:
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            logits = trainer.model(**inputs).logits
        return logits.argmax(dim=1).cpu().tolist()

    transformer_inference_time = (
        time_inference(transformer_predict, validation_texts) * 1000
    )
    results: list[dict[str, float | str]] = [
        {
            "Model": "TF-IDF + LogisticRegression",
            "Accuracy": tfidf_accuracy,
            "Macro F1": tfidf_macro_f1,
            "Training time (s)": tfidf_training_time,
            "Inference time (ms/text)": tfidf_inference_time,
        },
        {
            "Model": "Fine-tuned DistilBERT",
            "Accuracy": transformer_accuracy,
            "Macro F1": transformer_macro_f1,
            "Training time (s)": transformer_training_time,
            "Inference time (ms/text)": transformer_inference_time,
        },
    ]
    print("\n| Model | Accuracy | Macro F1 | Training time (s) | Inference time (ms/text) |")
    print("|---|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['Model']} | {result['Accuracy']:.3f} "
            f"| {result['Macro F1']:.3f} | {result['Training time (s)']:.2f} "
            f"| {result['Inference time (ms/text)']:.2f} |"
        )
    save_comparison_and_submission(results)
    save_api_screenshot()


if __name__ == "__main__":
    main()
