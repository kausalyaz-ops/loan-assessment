import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


SUPPORT_TICKETS = [
    "I was charged twice for my subscription",
    "The app crashes when I open settings",
    "Please reset my password, cannot log in",
    "Refund not received for cancelled order",
    "Error 500 when uploading a file",
    "How do I change my registered email",
]

LABELED_TICKETS = [
    ("I was charged twice for my subscription", "billing"),
    ("Refund not received for cancelled order", "billing"),
    ("Need an invoice for my monthly plan", "billing"),
    ("Payment failed but money was deducted", "billing"),
    ("Please cancel my plan and refund the charge", "billing"),
    ("Why was my card billed two times", "billing"),
    ("I was charged an extra fee on my account", "billing"),
    ("My order refund status is still pending", "billing"),
    ("Subscription renewal charged unexpectedly", "billing"),
    ("Requesting a refund for a duplicate payment", "billing"),
    ("The app crashes when I open settings", "technical"),
    ("Error 500 when uploading a file", "technical"),
    ("Cannot upload documents from my phone", "technical"),
    ("The screen freezes after the app update", "technical"),
    ("Website shows a server error when saving", "technical"),
    ("Settings page closes unexpectedly", "technical"),
    ("File upload is stuck at 90 percent", "technical"),
    ("Cannot open the dashboard because of an error", "technical"),
    ("The app crashes during sign in", "technical"),
    ("Upload fails with a network error", "technical"),
    ("Please reset my password, cannot log in", "account"),
    ("How do I change my registered email", "account"),
    ("I am locked out of my account", "account"),
    ("Password reset email never arrived", "account"),
    ("Update the email address on my profile", "account"),
    ("I forgot my password and cannot sign in", "account"),
    ("Change the email used to log in", "account"),
    ("Unable to access account after password change", "account"),
    ("Need help recovering my username", "account"),
    ("Two factor code is not working for login", "account"),
]


def tokenize(text: str) -> list[str]:
    """Lowercase text, remove punctuation, and split it into word tokens."""
    cleaned_text = re.sub(r"[^\w\s]", "", text.lower())
    return cleaned_text.split()


def main() -> None:
    cleaned_tickets = [" ".join(tokenize(ticket)) for ticket in SUPPORT_TICKETS]
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
    )
    ticket_features = vectorizer.fit_transform(cleaned_tickets)
    feature_names = vectorizer.get_feature_names_out()

    print(f"TF-IDF feature shape: {ticket_features.shape}")

    first_ticket_scores = ticket_features[0].toarray().ravel()
    top_word_indices = sorted(
        range(len(feature_names)),
        key=lambda index: (-first_ticket_scores[index], feature_names[index]),
    )[:5]
    top_words = [
        (feature_names[index], first_ticket_scores[index])
        for index in top_word_indices
    ]
    print(f"Top 5 words in ticket 1: {top_words}")

    query = "cannot log in to my account"
    query_features = vectorizer.transform([" ".join(tokenize(query))])
    similarities = cosine_similarity(query_features, ticket_features).ravel()
    most_similar_index = int(similarities.argmax())
    print(
        f"Most similar ticket: {most_similar_index + 1} "
        f"(cosine similarity: {similarities[most_similar_index]:.3f})"
    )
    print(f"Ticket text: {SUPPORT_TICKETS[most_similar_index]}")

    texts, labels = zip(*LABELED_TICKETS)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    classifier = TfidfVectorizer()
    train_features = classifier.fit_transform(train_texts)
    test_features = classifier.transform(test_texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    print(f"LogisticRegression accuracy: {accuracy_score(test_labels, predictions):.2f}")

    print("\nTF-IDF explanation:")
    print("TF-IDF scores how important a word is to a document.")
    print("It increases for words frequent in one document.")
    print("It decreases for words common across many documents.")


if __name__ == "__main__":
    main()
