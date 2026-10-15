from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "glove-wiki-gigaword-50"
OUTPUT_IMAGE = Path(__file__).with_name("word_vectors_pca.png")

PLOT_WORDS = [
    "king", "queen", "man", "woman", "prince", "princess", "boy", "girl",
    "father", "mother", "brother", "sister", "uncle", "aunt", "son",
    "daughter", "husband", "wife", "computer", "keyboard", "software",
    "internet", "phone", "technology", "car", "bus", "train", "airplane",
    "city", "country",
]


def sentence_vector(sentence: str, model: KeyedVectors) -> np.ndarray:
    """Average the pretrained vectors for words present in the vocabulary."""
    words = sentence.lower().split()
    known_vectors = [model[word] for word in words if word in model]
    if not known_vectors:
        raise ValueError(f"No words from this sentence are in the model: {sentence}")
    return np.mean(known_vectors, axis=0)


def main() -> None:
    print(f"Loading {MODEL_NAME} (downloads it the first time)...")
    word_vectors = api.load(MODEL_NAME)

    for word in ("king", "computer"):
        neighbours = word_vectors.most_similar(word, topn=5)
        print(f"\n5 nearest words to {word}: {neighbours}")

    analogy = word_vectors.most_similar(
        positive=["king", "woman"],
        negative=["man"],
        topn=1,
    )[0]
    print(f"\nking - man + woman -> {analogy[0]} (similarity: {analogy[1]:.3f})")

    sentence_a = "The king rules the country"
    sentence_b = "The queen leads the nation"
    vector_a = sentence_vector(sentence_a, word_vectors)
    vector_b = sentence_vector(sentence_b, word_vectors)
    similarity = float(cosine_similarity([vector_a], [vector_b])[0, 0])
    print(f"\nSentence 1: {sentence_a}")
    print(f"Sentence 2: {sentence_b}")
    print(f"Sentence cosine similarity: {similarity:.3f}")

    available_words = [word for word in PLOT_WORDS if word in word_vectors]
    if len(available_words) != len(PLOT_WORDS):
        missing_words = sorted(set(PLOT_WORDS) - set(available_words))
        raise ValueError(f"Words required for the PCA plot are missing: {missing_words}")

    vectors = np.array([word_vectors[word] for word in available_words])
    coordinates = PCA(n_components=2).fit_transform(vectors)
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.scatter(coordinates[:, 0], coordinates[:, 1])
    for word, (x_coordinate, y_coordinate) in zip(available_words, coordinates):
        axis.annotate(
            word,
            (x_coordinate, y_coordinate),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_title("GloVe Word Vectors Projected with PCA")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    figure.tight_layout()
    figure.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close(figure)
    print(f"\nPCA plot ({len(available_words)} words) saved to: {OUTPUT_IMAGE}")

    print("Embeddings represent words as dense vectors learned from their context.")
    print("Words used in similar contexts tend to have nearby vectors.")
    print("They capture semantic relationships, even when words do not exactly match.")
    print("TF-IDF mostly weights words by frequency and document rarity.")


if __name__ == "__main__":
    main()
