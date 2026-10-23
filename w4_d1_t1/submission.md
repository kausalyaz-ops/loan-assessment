# Foundation Models Comparison

*Snapshot: 23 October 2026. Context windows and capabilities vary by model/version; examples below are specific models, not guarantees for every model in a family.*

| Model family (example) | Owner | Open or closed weights | Context window | Strengths | Typical use |
|---|---|---|---|---|---|
| GPT (GPT-5.4) [1] | OpenAI | Closed; GPT-5.4 is accessed through OpenAI services, not downloadable weights. | 1.05M tokens | Complex professional work, reasoning, coding, image input and tool use. | Business assistants, coding, research and workflow automation. |
| Claude (Sonnet 5.5) [2] | Anthropic | Closed; accessed through Anthropic and partner services. | 1M tokens | Fast, capable text and image understanding, multilingual responses and tool use. | Customer support, writing, analysis and agentic workflows. |
| Llama (Llama 4 Scout / Llama 3.1 8B) [3] | Meta | Open weights under Meta's custom community license; use is subject to its terms. | Scout: 10M; Llama 3.1 8B: 128K tokens | Downloadable weights, local deployment, multilingual text; Scout also supports image input. | Private/local assistants, document search and custom applications. |
| Gemini (Gemini 4 Argon) [4][5] | Google | Closed; offered through Google services. | 1M tokens shared across input, thinking and output. | Multimodal understanding (text, image, audio and video) and very long-context work. | Document/media analysis, research and multimodal applications. |

## Two key terms (5 lines)
1. **Pre-trained on massive text** means the model learns statistical patterns from a very large collection of text before being adapted for particular uses.
2. During training, it repeatedly predicts missing or next tokens and adjusts internal parameters when its predictions are wrong.
3. The training data gives it broad language patterns and information, but it does not make every answer factual or current.
4. A **foundation model** is a broadly capable model trained on large, diverse data that can be adapted to many downstream tasks.
5. It can be used directly with prompts, tools or supplied context, or further adapted with fine-tuning.

## Five tasks without retraining
- Answer questions and explain concepts.
- Summarize or rewrite text.
- Translate between languages.
- Classify, extract, or organize information.
- Draft content, including code, from instructions.

## Recommendations
- **Customer-support bot — Claude Sonnet 5.5:** its speed/intelligence balance, multilingual support and tool use suit conversational support and integrations such as order lookup; connect it to approved support knowledge and evaluate responses before launch.
- **Private on-premise document tool — Llama 3.1 8B Instruct:** Meta provides downloadable weights and documents local use; its 128K context and smaller size make it a more practical on-premise starting point than the larger Llama 4 variants. Quantization may help fit local hardware. Check Meta's license and validate answer quality and hardware needs before deployment.

## Official sources
1. OpenAI, [GPT-5.4 model documentation](https://developers.openai.com/api/docs/models/gpt-5.4).
2. Anthropic, [Claude models overview](https://platform.claude.com/docs/en/models/overview) and [Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).
3. Meta, [Llama 4 Scout model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) and [Llama 3.1 8B Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
4. Google, [Gemini API models](https://ai.google.dev/gemini-api/docs/models).
5. Google, [Gemini long-context documentation](https://ai.google.dev/gemini-api/docs/long-context).
