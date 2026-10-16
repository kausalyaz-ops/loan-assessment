# BERT vs GPT

## BERT fill-mask predictions

- **The capital of France is [MASK].** -> `paris`
- **A dog is a type of [MASK].** -> `dog`
- **She wrote a letter with a [MASK].** -> `friend`

## GPT-2 generated continuations

- **Prompt:** The benefits of regular exercise include
  **Continuation:** :

Improved mood

Improved body image

Improved memory

Staying fit

Improved concentration

It's hard to believe
- **Prompt:** A good customer support reply should
  **Continuation:** be as simple as: "Hi, I am a customer support rep for this company. I have been asked by a customer support representative to look into
- **Prompt:** Artificial intelligence is changing education because
  **Continuation:** it is becoming a way of life for many.

The only way to ensure that this change is not an accident is to start with a system

## Parameter counts

- `bert-base-uncased`: 109,514,298 parameters
- `gpt2`: 124,439,808 parameters

## Comparison

| Aspect | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only Transformer | Decoder-only Transformer |
| Reads / writes | Reads both left and right context; fills a masked token | Reads preceding context; generates text from left to right |
| Best for | Understanding and classifying text | Generating coherent text continuations |
| Example tasks | Ticket classification, sentiment analysis, masked-word prediction | Drafting replies, creative writing, text completion |

## Why choose each model?

BERT is well suited to ticket classification because its encoder uses context from both sides of each word.
It produces contextual representations that help distinguish ticket intent and category.
GPT is well suited to drafting replies because its decoder generates fluent text one token at a time.
Given a ticket and instructions, GPT can continue the prompt with a relevant, natural-language response.
