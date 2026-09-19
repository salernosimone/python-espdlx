# espdlx

Train models in PyTorch and export them to `.espdl` — the native model format for ESP32 boards.

`espdlx` is the Python training side: build models from `espdlx.layers` inside an
`espdlx.Model`, train with your own PyTorch loop, then export via the standard path:
`torch.onnx.export` -> esp-dl quantization (`espdl_quantize_onnx` -> `.espdl`).

Deployment is paired with the **[espdlx Arduino library](https://github.com/salernosimone/arduino-espdlx)**
(esp-dl port for ESP32-S3, installable from the Arduino Library Manager) for maximum
ease of use: `convert()` emits an Arduino-ready `.h` header next to the `.espdl` file —
include it, load it with `dl::Model`, call `run()`.

## Supported layers

`espdlx.layers` is deliberately a **subset**: each layer is constrained to what esp-nn
implements as an efficient SIMD kernel on the ESP32-S3. Otherwise one could just use
plain PyTorch — the constraints are the point: if it trains here, it runs fast there.

| Layer | Constraint (ESP32-S3 SIMD) |
|---|---|
| `Conv2d` | `groups=1`, `dilation=1`, symmetric (`SAME`) padding on odd kernels |
| `DepthwiseConv2d` | `out_channels = in_channels × multiplier`, `dilation=1` |
| `Linear` | any dims (the old esp-nn %8 rule is gone — `Linear(12, 5)` proven on-device) |
| `Gemm` | ONNX semantics `alpha·A·B + beta·C`, `transB`, `transA` rejected; non-plain `alpha`/`beta` fold into weights at export |
| `Concat` | multi-input in `Model` via `input_indices=[...]` + `dim`; same non-axis dims required (dense blocks / FPN fusions) |
| `Resize` | nearest / bilinear / bicubic (`scale_factor` or `size`); the sanctioned upsample path (no `ConvTranspose` in the registry) |
| `MaxPool2d` / `AvgPool2d` | standard windows/strides |
| `ReLU` / `ReLU6` / `HardSwish` / `Sigmoid` / `Softmax` | plain torch semantics (`ReLU6` rewrites to `Clip(0, 6)` at export) |
| `LeakyReLU` / `Tanh` / `Swish` / `Elu` / `HardSigmoid` / `Clip` | device-proven (runtime `dl::LeakyRelu/Tanh/Swish/Elu/HardSigmoid/Clip`; `Swish` exports as `Sigmoid + Mul` at opset 13) |
| `BatchNorm2d` | standard torch semantics, folds at quantization |
| `Add` / `Sub` | constant, skip connection (`input_index=i`), or binary (`Add()`/`Sub()(x, y)`); same-shape tensors |
| `Mul` / `Div` | same three modes as `Add`/`Sub`; `Div` constant must be non-zero |
| `Neg` / `Exp` / `Log` / `Sqrt` | elementwise unary (`Log`/`Sqrt` dequantize to F32 at quantization; feed positive inputs) |
| `Mean` / `Flatten` | global average pool / row-major flatten |
| `MatMul` | binary, skip (`input_index=i`), or saved pair (`input_indices=[i, j]`) — `Q @ K^T`, `weights @ V`; rank ≥ 2, compose transposes via `Transpose` |
| `LayerNorm` / `RMSNorm` | feature-axis norms; `LayerNormalization` native, `RMSNorm` composite fuses at quantization (Tier-3 models export at opset 18) |
| `Transpose` / `Reshape` / `Squeeze` / `Unsqueeze` | full-perm transpose; per-sample reshape (batch-agnostic `-1`); size-1 drop/insert |
| `Slice` / `Gather` / `Pad` | static windows (ONNX clamp); frozen-index select; constant pad — use `Slice` (not `Split`) inside `Model` |
| `Split` | manual wiring only (tuple output); exports native `Split` at opset 18 |

Violations fail fast at construction time (`ValueError`), not on the board.

Full op coverage vs the esp-dl registry and proof status per op (Tier 3:
transformer path + shape ops device-proven (chain level, trained-like magnitudes); RNNs deferred):
see `docs/OPS_ROADMAP.md`.

## Installation

```bash
pip install espdlx
# with esp-dl quantization support:
pip install "espdlx[convert]"
```

## Example

Simple CNN trained on MNIST:

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import espdlx
import espdlx.layers as L

device = torch.device("mps")

model = espdlx.Model(
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

## Export to `.espdl`

Quantize a trained model to `.espdl` (needs the `convert` extra):

```python
from torch.utils.data import DataLoader, TensorDataset
from espdlx.convert import convert

calib_loader = DataLoader(TensorDataset(calib_images), batch_size=8)

report = convert(
    model=model.cpu(),              # any nn.Module; espdlx.Model or plain torch
    example_input=example_input,    # dummy input for torch.onnx.export
    calib_loader=calib_loader,      # calibration batches
    out_dir="deploy/mnist_cnn",       # receives .espdl + .onnx + .h + report
    calib_steps=16,
)
print(report["espdl_bytes"], report["input_scale"], report["input_zero_point"])
print(report["header_path"])  # deploy/mnist_cnn/mnist_cnn.h — drop next to your .ino
```

## Deploy with the espdlx Arduino library

Install `espdlx` from the Arduino Library Manager (ESP32-S3), then copy the generated
header (`deploy/mnist_cnn/mnist_cnn.h`) next to your sketch — no `xxd` needed,
`convert()` already emitted the model as a C array.

Run inference with `dl::Model` (pattern from the espdlx Arduino library examples):

```cpp
#include <Arduino.h>
#include <espdlx.h>
#include "mnist_cnn.h"  // const unsigned char mnist_cnn_espdl[] = { 0x45, 0x44, ... };

dl::Model *model = nullptr;

void setup() {
  Serial.begin(115200);

  // Load the .espdl model straight from flash.
  model = new dl::Model((const char *)mnist_cnn_espdl,
                        fbs::MODEL_LOCATION_IN_FLASH_RODATA);

  auto inputs = model->get_inputs();
  auto outputs = model->get_outputs();
  dl::TensorBase *in = inputs.begin()->second;
  dl::TensorBase *out = outputs.begin()->second;
  Serial.printf("in_elems=%d out_elems=%d\n", in->size, out->size);

  memcpy(in->data, my_input, in->size);  // int8 NHWC, quantized with input_scale
  model->run();                          // or model->run(dl::RUNTIME_MODE_MULTI_CORE)

  float *logits = (float *)out->data;
  int pred = 0;
  for (int i = 1; i < out->size; i++)
    if (logits[i] > logits[pred]) pred = i;
  Serial.printf("pred=%d\n", pred);
}

void loop() { delay(1000); }
```

The library also ships a `SelfTest` example (`examples/SelfTest/SelfTest.ino`) that
exercises core + vision + audio through the single `espdlx.h` header:

```cpp
#include <Arduino.h>
#include <espdlx.h>

void setup() {
  Serial.begin(115200);
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  dl::TensorBase t({4}, data, 0, dl::DATA_TYPE_FLOAT, true);
  dl::math::softmax((float *)t.data, 4);  // core check via espdlx.h
}

void loop() { delay(5000); }
```

## Model zoo

Prebuilt architectures (`espdlx/zoo.py`). The zoo keeps residual-free variants
for minimum latency, but the framework supports skip connections via
`Add(input_index=i)` (verified against the espdlx Arduino runtime's `dl::Add`):

```python
from espdlx.zoo import MobileNetV2, MobileNetV1, VGG, DSCNN

model = MobileNetV2.Slim(num_classes=5)  # flowers, 128px RGB
model = MobileNetV2.Base(num_classes=32) # CIFAR-100 pretrain backbone, 32px
model = MobileNetV2.Fomo()               # EXPERIMENTAL centroid detector, 96px
```

| Model | Input | Params | MMACs | `.espdl` | Sketch flash | PSRAM ctx | S3 latency single / multi |
|---|---|---:|---:|---:|---:|---:|---|
| `MobileNetV2.Slim(5)` | 128px RGB | 13,880 | 14.2 | 43.6 KB | 1.20 MB | 324 KB | 87 ms / 73 ms |
| `MobileNetV2.Base(5)` | 128px RGB | 132,280 | 100.2 | 169 KB | 1.33 MB | 1.09 MB | 243 ms / 256 ms |
| `MobileNetV2.Base(32)` | 32px RGB | 134,224 | 6.3 | 170 KB | 1.28 MB | 218 KB | 15.0 ms / 14.9 ms |
| `MobileNetV2.Fomo` (experimental) | 96px gray | 6,498 | 2.3 | 34.5 KB | 1.15 MB | 159 KB | 37.3 ms / 30.9 ms |
| `MobileNetV1.Slim(10)` | 96px RGB | 352,330 | 52.0 | 367 KB | 1.51 MB | 612 KB | 446 ms / 256 ms |
| `ResNet.BottleneckTiny(10)` | 96px RGB | 1,305,258 | 104.6 | 1.27 MB | 2.46 MB | 1.53 MB | 1045 ms / 811 ms |
| `VGG.Small(10)` | 96px RGB | 295,770 | 120.8 | 303 KB | 1.44 MB | 624 KB | 716 ms / 638 ms |
| `VGG.Stride(10)` | 96px RGB | 1,219,450 | 174.2 | 1.18 MB | 2.37 MB | 1.53 MB | 1570 ms / 1296 ms |
| `VGG.Wide(10)` | 96px RGB | 1,560,522 | 212.7 | 1.50 MB | 2.70 MB | 1.72 MB | 3102 ms / 2501 ms |
| `VGG.Base(10)` | 96px RGB | 663,106 | 233.0 | 662 KB | 1.81 MB | 973 KB | 2297 ms / 1780 ms |
| `VGG.Large(10)` | 96px RGB | 1,176,746 | 475.2 | 1.14 MB | 2.32 MB | 1.80 MB | 4804 ms / 3560 ms |
| `DSCNN.Tiny(2)` | 40x32 mel | 968 | 0.8 | 6.8 KB | 1.11 MB | 66 KB | 26.3 ms / 17.0 ms |
| `DSCNN.Small(2)` | 40x98 mel | 3,160 | 1.7 | 16.2 KB | 1.12 MB | 93 KB | 16.0 ms / 14.2 ms |

- All device numbers measured 2026-09-12 on ESP32-S3 @ 240 MHz via esp-dl
  (`.espdl` w8a8, OPI PSRAM, `huge_app`, N=20 runs, `micros()` around `model->run()`).
- `.espdl` from `espdlx.convert` after 1 epoch of synthetic-noise training per model;
  sizes and latency are weights-independent (accuracy on noise weights is meaningless,
  so no accuracy column).
- PSRAM ctx = model context + working set (`free_psram` drop after load).
  Sketch flash is ~1.1 MB framework floor + model + test vector.
- `MobileNetV2.Fomo` / `espdlx.fomo` are **experimental**: synthetic unit tests
  only, no on-device accuracy verification yet (latency above just proves it
  runs). Best real-data run so far: P=0.03/R=0.28 — not a working detector yet.

## License

MIT — see `LICENSE`.
