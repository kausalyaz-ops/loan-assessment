from functools import lru_cache
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(__file__).with_name("fine_tuned_distilbert")
MAX_LENGTH = 64
app = FastAPI(title="DistilBERT Sentiment API")


class PredictRequest(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    @field_validator("text")
    @classmethod
    def ensure_non_empty_text(cls, value: str) -> str:
        if not value:
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_dir}. "
                "Run transformers_training_and_serving.py first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> dict[str, str | float]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            probabilities = self.model(**encoded).logits.softmax(dim=-1)[0]
        predicted_id = int(probabilities.argmax().item())
        return {
            "label": str(self.id2label[predicted_id]),
            "confidence": float(probabilities[predicted_id].item()),
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, str | float]:
    return get_predictor().predict(request.text)
