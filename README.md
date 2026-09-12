# espnnpy

esp-nn friendly NN blocks for ESP32-S3 (PyTorch backend).

Build models from `espnn.layers` inside an `espnn.Model`, train with your own
PyTorch loop, then deploy via the standard path: `torch.onnx.export` ->
esp-dl quantization (`espdl_quantize_onnx` -> `.espdl`).

## Installation

```bash
pip install espnnpy
```

## Example

Simple CNN trained on MNIST:

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import espnn
import espnn.layers as L

device = torch.device("mps")

model = espnn.Model(
    [
        L.Conv2d(1, 16, 3, padding=1), L.ReLU(),
        L.MaxPool2d(2),
        L.Conv2d(16, 32, 3, padding=1), L.ReLU(),
        L.MaxPool2d(2),
        L.Conv2d(32, 10, 1),
        L.Mean(),
        L.Flatten(),
    ],
    name="mnist_cnn",
).to(device)

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
train_ld = DataLoader(train_ds, batch_size=128, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=512)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for epoch in range(3):
    for x, y in train_ld:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_ld:
        correct += (model(x.to(device)).argmax(-1) == y.to(device)).sum().item()
        total += len(y)
print(f"test acc: {correct / total:.3f}")
```

## License

MIT — see `LICENSE`.
