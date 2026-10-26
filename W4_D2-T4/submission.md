# Transformers vs RNNs

## Toy sequence task and results

Both models were trained to classify length-12 digit sequences as nondecreasing (sorted) or not sorted. The balanced dataset pairs each sorted sequence with an unsorted version that has the same digits, split into 1024 training and 256 held-out examples. Each model was trained for 100 epochs with the same batch size and optimizer.

| Model | Held-out accuracy | Training time (seconds) |
|---|---:|---:|
| LSTM | 97.7% | 4.27 |
| TransformerEncoder (2 layers) | 72.7% | 15.97 |

Training time is measured on the machine running this script and includes training epochs only, not model initialization or test evaluation. Results are for this small synthetic task; they do not represent general benchmark rankings.

## Four reasons Transformers replaced RNNs

1. Parallelism: unlike a recurrent network, a Transformer can process all input positions together during training, making better use of modern accelerators.
2. Long-range attention: self-attention can directly connect distant tokens instead of passing information step by step through a recurrent state.
3. Scaling: the architecture scales effectively with larger datasets, model sizes, and compute, enabling strong general-purpose capabilities.
4. Transfer learning: pretrained Transformer models can be adapted to many downstream tasks, reducing the need to train each model from scratch.

## Vanishing gradients in four lines

During backpropagation, gradients are multiplied across many time steps.
If those factors are repeatedly smaller than one, the gradient shrinks toward zero.
Early sequence steps then receive almost no learning signal.
The network struggles to learn relationships between events far apart in the sequence.

## How ChatGPT-style models generate text

ChatGPT generates a response by repeatedly predicting what should come next. First, it breaks your prompt into pieces called tokens, which may be words or parts of words. It turns those tokens into numbers and uses the surrounding context to estimate a probability for many possible next tokens. For example, after 'The meeting starts at,' it may consider times or other likely continuations. The system selects one candidate, then adds it to the text already written. It uses the expanded text to make another prediction, repeating this process until it reaches a stopping point or a length limit. The model does not plan every sentence in advance; each choice influences later choices. Instructions and conversation history help shape those probabilities, so the same model can answer questions, draft messages, or summarize documents. Its fluency does not guarantee correctness: it can produce confident mistakes, so important claims still need human review.

*Word count: 150.*

## Bonus: KV caching

During autoregressive generation, the model computes key and value vectors for each new token in every attention layer. KV caching stores those vectors so they do not need to be recomputed at every next-token step. The model processes only the newest token against the cached history, reducing repeated computation and speeding up long responses. The cache uses additional memory, which grows with sequence length.
