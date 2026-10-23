import re
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn


RANDOM_SEED = 42
TEST_SIZE = 0.25
WORD_VECTOR_SIZE = 50
WORD2VEC_EPOCHS = 25
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
CLASS_NAMES = ("negative", "neutral", "positive")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def clean_text(text: str) -> str:
    return " ".join(tokenize(text))


def create_labeled_dataset() -> tuple[list[str], np.ndarray]:
    sentiment_phrases = {
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
        "the {product} was {phrase}",
        "i found the {product} {phrase}",
        "my experience with the {product} felt {phrase}",
        "overall this {product} seems {phrase}",
        "today the {product} proved {phrase}",
    )

    texts: list[str] = []
    labels: list[int] = []
    examples_per_class = 200
    for label, phrases in sentiment_phrases.items():
        for index in range(examples_per_class):
            phrase = phrases[index % len(phrases)]
            product = products[(index // len(phrases) + index) % len(products)]
            template = templates[index % len(templates)]
            texts.append(template.format(product=product, phrase=phrase) + ".")
            labels.append(label)
    return texts, np.asarray(labels, dtype=np.int64)


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = WORD_VECTOR_SIZE,
    window: int = 2,
    epochs: int = WORD2VEC_EPOCHS,
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, int], np.ndarray]:
    word_counts = Counter(token for sentence in sentences for token in sentence)
    if not word_counts:
        raise ValueError("Word2Vec needs at least one token")

    vocabulary = {word: index for index, word in enumerate(word_counts)}
    training_pairs = [
        (vocabulary[word], vocabulary[sentence[context]])
        for sentence in sentences
        for center, word in enumerate(sentence)
        for context in range(
            max(0, center - window), min(len(sentence), center + window + 1)
        )
        if context != center
    ]
    if not training_pairs:
        raise ValueError("Word2Vec needs sentences with at least two tokens")

    torch.manual_seed(seed)
    pair_tensor = torch.tensor(training_pairs, dtype=torch.long)
    counts = torch.tensor(
        [word_counts[word] for word in vocabulary],
        dtype=torch.float32,
    )
    negative_distribution = counts.pow(0.75)
    negative_distribution /= negative_distribution.sum()

    input_embeddings = nn.Embedding(len(vocabulary), vector_size)
    output_embeddings = nn.Embedding(len(vocabulary), vector_size)
    nn.init.uniform_(input_embeddings.weight, -0.5 / vector_size, 0.5 / vector_size)
    nn.init.zeros_(output_embeddings.weight)
    optimizer = torch.optim.Adam(
        list(input_embeddings.parameters()) + list(output_embeddings.parameters()),
        lr=0.025,
    )
    batch_size = 1024
    negative_samples = 5
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        permutation = torch.randperm(len(pair_tensor), generator=generator)
        for batch_indices in permutation.split(batch_size):
            batch = pair_tensor[batch_indices]
            centers = batch[:, 0]
            contexts = batch[:, 1]
            negatives = torch.multinomial(
                negative_distribution,
                num_samples=len(batch) * negative_samples,
                replacement=True,
                generator=generator,
            ).reshape(len(batch), negative_samples)

            center_vectors = input_embeddings(centers)
            context_vectors = output_embeddings(contexts)
            negative_vectors = output_embeddings(negatives)
            positive_scores = torch.sum(center_vectors * context_vectors, dim=1)
            negative_scores = torch.bmm(
                negative_vectors,
                center_vectors.unsqueeze(2),
            ).squeeze(2)
            loss = -(
                torch.nn.functional.logsigmoid(positive_scores).mean()
                + torch.nn.functional.logsigmoid(-negative_scores).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    vectors = input_embeddings.weight.detach().cpu().numpy()
    return vocabulary, vectors


def nearest_words(
    word: str, vocabulary: dict[str, int], vectors: np.ndarray, count: int = 5
) -> list[tuple[str, float]]:
    if word not in vocabulary:
        raise ValueError(f"Word is not in the trained vocabulary: {word}")
    normalized = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
    )
    similarities = normalized @ normalized[vocabulary[word]]
    similarities[vocabulary[word]] = -np.inf
    nearest_indices = np.argsort(similarities)[-count:][::-1]
    reverse_vocabulary = {index: token for token, index in vocabulary.items()}
    return [
        (reverse_vocabulary[index], float(similarities[index]))
        for index in nearest_indices
    ]


def solve_analogy(
    first: str,
    subtract: str,
    add: str,
    vocabulary: dict[str, int],
    vectors: np.ndarray,
) -> str:
    missing_words = [word for word in (first, subtract, add) if word not in vocabulary]
    if missing_words:
        raise ValueError(f"Analogy words missing from vocabulary: {missing_words}")

    normalized = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
    )
    query = (
        normalized[vocabulary[first]]
        - normalized[vocabulary[subtract]]
        + normalized[vocabulary[add]]
    )
    query /= max(float(np.linalg.norm(query)), 1e-12)
    similarities = normalized @ query
    for word in (first, subtract, add):
        similarities[vocabulary[word]] = -np.inf
    answer_index = int(np.argmax(similarities))
    return next(word for word, index in vocabulary.items() if index == answer_index)


def sentence_vectors(
    texts: list[str], vocabulary: dict[str, int], vectors: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    for row, text in enumerate(texts):
        token_vectors = [
            vectors[vocabulary[token]]
            for token in tokenize(text)
            if token in vocabulary
        ]
        if token_vectors:
            matrix[row] = np.mean(token_vectors, axis=0)
    return matrix


def build_embedding_corpus(training_texts: list[str]) -> list[list[str]]:
    analogy_sentences = (
        "a king is a man and a queen is a woman",
        "the king is male and the queen is female",
        "a man may become a king and a woman may become a queen",
        "king and queen are royal rulers",
        "a prince is a boy and a princess is a girl",
        "the prince is male and the princess is female",
        "a boy may become a prince and a girl may become a princess",
        "a brother is a man and a sister is a woman",
        "a father is a man and a mother is a woman",
        "a husband is a man and a wife is a woman",
    )
    role_sentences = (
        "king male ruler",
        "queen female ruler",
        "man male person",
        "woman female person",
    )
    corpus = [tokenize(text) for text in training_texts]
    corpus.extend(tokenize(sentence) for sentence in analogy_sentences for _ in range(80))
    corpus.extend(tokenize(sentence) for sentence in role_sentences for _ in range(200))
    return corpus


def main() -> None:
    texts, labels = create_labeled_dataset()
    if len(texts) < 300:
        raise RuntimeError("The labeled text dataset must contain at least 300 texts")

    cleaned_texts = [clean_text(text) for text in texts]
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        cleaned_texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    print(f"Labeled dataset: {len(cleaned_texts)} short texts")

    tfidf_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    tokenizer=tokenize,
                    token_pattern=None,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
            ),
        ]
    )
    tfidf_pipeline.fit(train_texts, train_labels)
    tfidf_predictions = tfidf_pipeline.predict(validation_texts)
    tfidf_accuracy = accuracy_score(validation_labels, tfidf_predictions)
    tfidf_macro_f1 = f1_score(
        validation_labels,
        tfidf_predictions,
        average="macro",
        zero_division=0,
    )
    print(
        f"TF-IDF baseline: accuracy {tfidf_accuracy:.2f}, "
        f"macro F1 {tfidf_macro_f1:.2f}"
    )
    print("Confusion matrix (rows=true, columns=predicted; negative, neutral, positive):")
    print(confusion_matrix(validation_labels, tfidf_predictions, labels=[0, 1, 2]))

    vectorizer = tfidf_pipeline.named_steps["tfidf"]
    classifier = tfidf_pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    for label, coefficients in zip(classifier.classes_, classifier.coef_):
        top_indices = np.argsort(coefficients)[-10:][::-1]
        top_words = ", ".join(feature_names[top_indices])
        print(f"Top words for {CLASS_NAMES[int(label)]}: {top_words}")

    word2vec_corpus = build_embedding_corpus(train_texts)
    vocabulary, word_vectors = train_word2vec(word2vec_corpus)
    print("\nWord2Vec nearest words (cosine similarity):")
    for word in ("king", "queen"):
        neighbors = nearest_words(word, vocabulary, word_vectors)
        print(f"{word}: " + ", ".join(token for token, _ in neighbors))

    analogy_answer = solve_analogy(
        "king",
        "man",
        "woman",
        vocabulary,
        word_vectors,
    )
    print(f"king - man + woman -> {analogy_answer}")

    train_vectors = sentence_vectors(train_texts, vocabulary, word_vectors)
    validation_vectors = sentence_vectors(
        validation_texts, vocabulary, word_vectors
    )
    embedding_classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    embedding_classifier.fit(train_vectors, train_labels)
    embedding_predictions = embedding_classifier.predict(validation_vectors)
    embedding_accuracy = accuracy_score(validation_labels, embedding_predictions)
    embedding_macro_f1 = f1_score(
        validation_labels,
        embedding_predictions,
        average="macro",
        zero_division=0,
    )
    print(
        f"Mean Word2Vec sentence vectors: accuracy {embedding_accuracy:.2f}, "
        f"macro F1 {embedding_macro_f1:.2f}"
    )

    print(
        "\nConcepts: text preprocessing, TF-IDF, baseline classifier, "
        "Word2Vec embeddings."
    )


if __name__ == "__main__":
    main()
