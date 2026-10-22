# Transformers: Pipelines, Fine-tuning and Serving

Dataset: generated short sentiment texts (60 examples per class). Metrics use a stratified 25% holdout.

## TF-IDF vs fine-tuned DistilBERT

| Model | Accuracy | Macro F1 | Training time (s) | Inference time (ms/text) |
|---|---:|---:|---:|---:|
| TF-IDF + LogisticRegression | 1.000 | 1.000 | 0.03 | 0.06 |
| Fine-tuned DistilBERT | 1.000 | 1.000 | 19.30 | 7.83 |

Training and inference times are measured on the machine running this script and are not directly comparable across hardware.

## BERT encoder vs GPT decoder (5 lines)

1. BERT is an encoder-only Transformer that reads context from both directions.
2. GPT is a decoder-only Transformer that predicts the next token from left context.
3. BERT is designed for understanding tasks such as classification and named-entity recognition.
4. GPT is designed for text generation, continuation, and conversational responses.
5. I would use DistilBERT for this sentiment task because it is compact and suited to classification.

## API response screenshot

See `api_response.png` for the captured successful POST `/predict` response.

## Run the API

Run `python transformers_training_and_serving.py` once to train and save the model, then start the service with `uvicorn api:app --host 127.0.0.1 --port 8000`.

Example request: `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"I love this app\"}"`.

Whitespace-only and empty text are rejected with HTTP 422.
