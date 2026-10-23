"""Count sample text tokens and estimate Claude API usage costs.

Token counts use tiktoken's cl100k_base encoding as a reproducible exercise.
Anthropic uses its own tokenizer, so these counts are illustrative rather than
an exact prediction of Claude API billing.
"""

import json

import tiktoken


ENCODING_NAME = "cl100k_base"
WORDS_PER_TOKEN_RULE_OF_THUMB = 0.75
TOKENS_PER_MILLION = 1_000_000

# Claude Sonnet 5.5 standard API rates, USD per million tokens, checked
# 2026-10-23: https://platform.claude.com/docs/en/about-claude/pricing
PRICE_IN_PER_MILLION = 2.00
PRICE_OUT_PER_MILLION = 10.00

CHATS_PER_DAY = 1_000
DAYS_PER_MONTH = 30
INPUT_TOKENS_PER_CHAT = 1_500
OUTPUT_TOKENS_PER_CHAT = 400

SAMPLE_TEXTS = {
    "Sentence": (
        "A customer asks how to update the delivery address for an order "
        "that has not shipped yet."
    ),
    "Paragraph": (
        "A helpful customer-support assistant should answer clearly, politely, "
        "and consistently. It should use the company knowledge base instead "
        "of guessing, ask a brief follow-up question when important information "
        "is missing, and explain the next step the customer can take. When a "
        "request involves private account details, the assistant should verify "
        "the customer through the approved process before sharing information "
        "or making changes. If it cannot resolve an issue, it should summarize "
        "the case and route it to a human agent."
    ),
    "Page": (
        "Managing a delivery: customer guide\n\n"
        "Before an order ships, customers can usually update the delivery "
        "address from the order details page. Sign in to the account used to "
        "place the order, open Order history, and select the order that needs "
        "a change. If an Edit delivery address option is available, enter the "
        "new address and save it. Check that the apartment, unit, postal code, "
        "and contact phone number are correct before confirming. The account "
        "page should show the updated address after the change is accepted.\n\n"
        "Address changes may not be available for every order. A warehouse "
        "can begin preparing a package soon after checkout, and a carrier may "
        "take control once a shipping label is created. In those cases, the "
        "website may no longer allow an edit. Contact customer support as soon "
        "as possible and provide the order number. For privacy, do not send "
        "full payment card details or account passwords in a message. Support "
        "may ask you to verify your identity through the account or another "
        "approved secure method.\n\n"
        "After dispatch, review the tracking page for options provided by the "
        "carrier. Some carriers allow a delivery instruction, a safe-place "
        "request, or collection from a nearby pickup point. These options "
        "depend on the service, destination, and parcel. The store cannot "
        "promise that a carrier will accept a late change, and a request may "
        "take time to appear in tracking. Keep the original delivery address "
        "available until the carrier confirms a change.\n\n"
        "If an address was entered incorrectly and the parcel cannot be "
        "rerouted, check the store's delivery and returns policy before "
        "placing another order. A returned parcel may need to be received and "
        "processed before a replacement or refund can be arranged. Keep "
        "confirmation emails and support case numbers so the team can find "
        "the conversation quickly. If an order is marked delivered but cannot "
        "be found, first check the mailbox, building reception, household "
        "members, and the carrier's delivery photo or notes. Then contact "
        "support with the tracking number and the date you checked.\n\n"
        "For urgent changes, use the support channel listed on the official "
        "store website. Explain what needs to change, but share only the "
        "information required to identify the order. A support assistant can "
        "explain available options and hand the request to a person when the "
        "change requires an exception. Availability, deadlines, and fees vary "
        "by store and carrier, so the order page and current policy are the "
        "best sources for the final answer."
    ),
    "JSON": json.dumps(
        {
            "ticket_id": "CS-10482",
            "customer": {"name": "Asha Kumar", "language": "en"},
            "issue": "delivery address update",
            "order_status": "processing",
            "shipped": False,
            "requested_action": "change address",
            "priority": "normal",
        },
        indent=2,
    ),
    "Hindi sentence": "नमस्ते, कृपया मेरे ऑर्डर की डिलीवरी की तारीख बताइए।",
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    price_in: float,
    price_out: float,
) -> float:
    """Return USD cost when prices are quoted per million tokens."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative.")
    if price_in < 0 or price_out < 0:
        raise ValueError("Token prices must be non-negative.")
    return (
        input_tokens * price_in + output_tokens * price_out
    ) / TOKENS_PER_MILLION


def main() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    print(f"Tokenizer: tiktoken {ENCODING_NAME}")
    print("Words are whitespace-separated; ratios are approximate.")
    print()

    counts: dict[str, int] = {}
    for label, text in SAMPLE_TEXTS.items():
        word_count = len(text.split())
        token_count = len(encoding.encode(text))
        counts[label] = token_count
        words_per_token = word_count / token_count if token_count else 0.0
        comparison = (
            "near the rule of thumb"
            if abs(words_per_token - WORDS_PER_TOKEN_RULE_OF_THUMB) <= 0.15
            else (
                "above the rule of thumb"
                if words_per_token > WORDS_PER_TOKEN_RULE_OF_THUMB
                else "below the rule of thumb"
            )
        )
        print(
            f"{label}: {word_count} words -> {token_count} tokens | "
            f"{words_per_token:.2f} words/token ({comparison})"
        )

    print()
    for context_size in (8_000, 128_000):
        fitting = [
            label
            for label, token_count in counts.items()
            if token_count <= context_size
        ]
        print(
            f"{context_size:,}-token context: "
            f"{', '.join(fitting) if fitting else 'none'} fit "
            "(each text considered separately)."
        )

    chats_per_month = CHATS_PER_DAY * DAYS_PER_MONTH
    monthly_input_tokens = chats_per_month * INPUT_TOKENS_PER_CHAT
    monthly_output_tokens = chats_per_month * OUTPUT_TOKENS_PER_CHAT
    monthly_cost = estimate_cost(
        monthly_input_tokens,
        monthly_output_tokens,
        PRICE_IN_PER_MILLION,
        PRICE_OUT_PER_MILLION,
    )

    print()
    print(
        "Monthly cost estimate: "
        f"{CHATS_PER_DAY:,} chats/day x {DAYS_PER_MONTH} days = "
        f"{chats_per_month:,} chats"
    )
    print(
        f"Input: {monthly_input_tokens:,} tokens at "
        f"${PRICE_IN_PER_MILLION:.2f}/million; "
        f"output: {monthly_output_tokens:,} tokens at "
        f"${PRICE_OUT_PER_MILLION:.2f}/million."
    )
    print(f"Estimated total: ${monthly_cost:,.2f} USD/month.")
    print(
        "Pricing source: https://platform.claude.com/docs/en/about-claude/pricing"
    )
    print(
        "Estimate excludes caching, tools, taxes, and tokenizer differences; "
        "actual API usage determines billing."
    )


if __name__ == "__main__":
    main()
