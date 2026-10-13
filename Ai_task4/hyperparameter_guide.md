# Hyperparameter guide

| Hyperparameter | If too low | If too high |
|---|---|---|
| Learning rate | Learning is very slow or stalls before a useful solution. | Updates overshoot; loss may oscillate or diverge. |
| Batch size | Noisy updates and slower training from frequent updates. | Smoother updates but more memory, fewer updates per epoch, and potentially poorer generalization. |
| Epochs | The model may underfit because training stops too soon. | Training takes longer and may overfit without early stopping or other regularization. |
| Weight decay | Little constraint on large weights; overfitting may increase. | Weights are shrunk too strongly; the model may underfit. |

Adam adapts each parameter's update using moving averages of gradients and squared gradients.
Adam with weight decay applies the decay through Adam's gradient update, which couples decay and adaptation.
AdamW decouples weight decay from the gradient update, making regularization more direct and predictable.
