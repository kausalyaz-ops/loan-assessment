import numpy as np
import torch


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tensor.shape}, "
        f"dtype={tensor.dtype}, device={tensor.device}"
    )


tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 3))
sequence = torch.arange(12)
random_tensor = torch.randn((2, 3))

print_tensor_info("torch.tensor", tensor)
print_tensor_info("torch.zeros", zeros)
print_tensor_info("torch.ones", ones)
print_tensor_info("torch.arange", sequence)
print_tensor_info("torch.randn", random_tensor)

matrix = torch.arange(1, 13).reshape(3, 4)
print("First row:", matrix[0, :])
print("First column:", matrix[:, 0])
print("Reshaped with view:", matrix.view(4, 3))
print("Reshaped with reshape:", matrix.reshape(2, 6))
print("Transposed:", matrix.transpose(0, 1))

left = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
right = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2)
product_with_operator = left @ right
product_with_matmul = torch.matmul(left, right)
print("(3x4) @ (4x2) ->", product_with_operator.shape)
print("torch.matmul result shape:", product_with_matmul.shape)

numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor_from_numpy = torch.from_numpy(numpy_array)
array_from_tensor = tensor_from_numpy.numpy()
print_tensor_info("torch.from_numpy", tensor_from_numpy)
print("NumPy array from tensor:\n", array_from_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_tensor = left.to(device)
if device.type == "cuda":
    print("Device: cuda")
else:
    print("Device: cpu")
print_tensor_info("Tensor on selected device", device_tensor)
