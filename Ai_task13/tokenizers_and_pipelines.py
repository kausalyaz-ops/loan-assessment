from transformers import AutoTokenizer, pipeline


TOKENIZER_MODEL = "distilbert-base-uncased"
TOKENIZER_SENTENCES = [
    "Hugging Face makes pretrained language models easy to use.",
    "Ravi works at Yoho Technologies in Chennai.",
    "The customer needs help with a recent payment.",
]
RARE_WORD = "antidisestablishmentarianism"
SENTIMENT_SENTENCES = [
    "I love how quickly the support team solved my problem.",
    "The new update is excellent and easy to use.",
    "My order arrived late and the package was damaged.",
    "The app keeps crashing whenever I try to sign in.",
    "The service was okay, but nothing special.",
]
NER_TEXT = (
    "Ravi from Yoho Technologies paid Rs. 45000 in Chennai on 12 March"
)
TICKET = "I was charged twice for my subscription and need a refund."
TICKET_LABELS = ["Billing", "Technical", "Account"]


def show_tokenizer_examples() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print("=== AutoTokenizer ===")
    for sentence in TOKENIZER_SENTENCES:
        encoded = tokenizer(sentence)
        print(f"\nSentence: {sentence}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
        print(f"input_ids: {encoded['input_ids']}")
        print(f"attention_mask: {encoded['attention_mask']}")

    rare_word_tokens = tokenizer.tokenize(RARE_WORD)
    print(f"\nRare word: {RARE_WORD}")
    print(f"Sub-word tokens: {rare_word_tokens}")

    print("\nBERT vs GPT:")
    print("BERT uses a bidirectional encoder to represent context on both sides.")
    print("GPT uses a causal decoder to predict the next token from prior tokens.")


def show_sentiment_pipeline() -> None:
    sentiment_classifier = pipeline("sentiment-analysis")
    print("\n=== Sentiment analysis pipeline ===")
    for sentence, result in zip(
        SENTIMENT_SENTENCES, sentiment_classifier(SENTIMENT_SENTENCES)
    ):
        print(
            f"{sentence}\n"
            f"  label: {result['label']}, score: {result['score']:.4f}"
        )


def show_ner_pipeline() -> None:
    ner = pipeline("ner", aggregation_strategy="simple")
    print("\n=== Named entity recognition pipeline ===")
    print(f"Text: {NER_TEXT}")
    for entity in ner(NER_TEXT):
        print(
            f"  {entity['word']} - {entity['entity_group']} "
            f"(score: {entity['score']:.4f})"
        )


def show_zero_shot_pipeline() -> None:
    classifier = pipeline("zero-shot-classification")
    result = classifier(TICKET, candidate_labels=TICKET_LABELS)
    print("\n=== Zero-shot classification pipeline ===")
    print(f"Ticket: {TICKET}")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score:.4f}")
    print(f"Predicted category: {result['labels'][0]}")


def main() -> None:
    show_tokenizer_examples()
    show_sentiment_pipeline()
    show_ner_pipeline()
    show_zero_shot_pipeline()


if __name__ == "__main__":
    main()
