import torch


w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
loss = (w * 2 + b - 10) ** 2
loss.backward()
print(f"w.grad = {w.grad.item()}, b.grad = {b.grad.item()}")

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.01

for step in range(30):
    loss = (w * 2 + b - 10) ** 2
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

print(
    f"After 30 gradient descent steps: "
    f"w = {w.item():.4f}, b = {b.item():.4f}, loss = {loss.item():.4f}"
)

w_accumulated = torch.tensor(1.0, requires_grad=True)
b_accumulated = torch.tensor(0.0, requires_grad=True)

for step in range(2):
    accumulated_loss = (w_accumulated * 2 + b_accumulated - 10) ** 2
    accumulated_loss.backward()
    print(
        f"Without zeroing, after backward {step + 1}: "
        f"w.grad = {w_accumulated.grad.item()}, "
        f"b.grad = {b_accumulated.grad.item()}"
    )

print("requires_grad=True tells autograd to track operations on a tensor.")
print("backward() computes gradients and adds them to each tensor's .grad.")
print("detach() returns a tensor disconnected from autograd's computation graph.")
print("Concepts: autograd, backward, grad accumulation, no_grad")
