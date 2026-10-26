# W2_D2_T2: Multi-Head Attention

Run `python multi_head_attention.py` to calculate two-head attention from
scratch and inspect a pretrained BERT model. The NumPy example projects 4D
token vectors, splits each projection into two 2D heads, computes scaled
dot-product attention per head, concatenates the head outputs, and applies an
output projection.

The Hugging Face example runs `bert-base-uncased` on
`The bank by the river was steep` with `output_attentions=True`. It prints
the last-layer attention distribution for the `bank` query in head 0 and
saves that head's full attention matrix as a token-labeled heatmap at
`attention_heatmap.png`.

## Why use several heads? (4 lines)

1. Different heads can focus on different relationships between tokens.
2. One head may capture nearby syntax while another tracks longer-range context.
3. Parallel heads let the model combine these distinct views in one layer.
4. Concatenating the heads preserves their information for the output projection.

## Decoder attention mask

A causal mask blocks each decoder token from attending to later tokens.
This prevents a prediction from seeing future words during autoregressive
training or generation; the model can attend only to the current and preceding
positions.
