# Ops roadmap — esp-dl registry vs `espdlx.layers`

Source of truth for device support: `register_module` list in the espdlx Arduino
library (`src/dl_module_creator.hpp`, esp-dl port). Runtime support alone is NOT
proof — every op needs the full Mul treatment before shipping:

1. unit parity + shapes (tests/)
2. ONNX export emits the node, `onnx.checker`-clean (tests/test_convert.py)
3. real `espdl_quantize_onnx` produces `.espdl` (no stubs)
4. runs on ESP32-S3 via espdlx (`dl::<Op>` resolves, clean `run()`)
5. accuracy MATCH vs float ref within ~2 LSB (like `flower_espdl.ino`)

Helper: `.scratch/check_luts.py` verifies PPQ-baked LUT tables against the
graph's actual tensor scales (lesson learned: always analyse the `.info` from
the EXACT convert run — PPQ scales vary run to run).

Proof sketches live in `.scratch/` (`muladd_proof.py`, `tier1_proof.py`) and
`Arduino/projects/TF/<name>_proof/` (git-ignored / external; methodology only).

## Tier 1 — SHIPPED

All of the below are device-proven in ONE composite chain
(`Arduino/projects/TF/tier1_proof`, not committed): Conv -> LeakyReLU -> Conv ->
Tanh -> Sub(skip) -> Swish -> Elu -> HardSigmoid -> Add(const) -> Log -> Exp ->
Sqrt -> Div(skip) -> Neg -> Clip, weights x0.15 (trained-like magnitudes),
in-calib vectors, 3/3 MATCH at maxabs 0.003. Individual sweeps confirmed
Swish/Elu/Neg/Clip at ~0.02 and tensor-Sub/Div at higher rate via the chain
(runtime `dl::Sub`/`dl::Div` scale-align inputs; standalone sweeps stress
sigmoid-saturated ranges).

`Add`, `Sub`, `Mul`, `Div` (constant / `input_index` skip / binary);
`Neg`, `Exp`, `Log`, `Sqrt`; `LeakyReLU`, `Tanh`, `Swish` (= `Sigmoid`+`Mul`
composite, no native SiLU at opset 13), `Elu`, `HardSigmoid`, `Clip`;
plus the pre-existing `Conv2d`, `DepthwiseConv2d`, `Linear`, `MaxPool2d`,
`AvgPool2d`, `ReLU`, `ReLU6`, `HardSwish`, `Sigmoid`, `Softmax`, `BatchNorm2d`
(folds at quantization), `Mean`, `Flatten`.

Relaxed rules (proven, were esp-nn-era legacy):
- `Linear` %8 in/out rule DROPPED — `Linear(12, 5)` runs 3/3 MATCH (~0.003).
  Zoo keeps `_pad8` heads as harmless legacy headroom.
- Still to verify: `Conv2d` dilation=1 / groups=1 rules against esp-dl `Conv`.

## Tier 2 — small design work (NOT started)

- `Concat` — needs multi-input `Model` support (`input_indices=[...]`, dim arg).
  Unlocks DenseNet / FPN-style nets.
- `Resize` (NEAREST / LINEAR / CUBIC all in runtime) — upsampling for
  detector/segmenter heads. Note: **no `ConvTranspose` in the registry**, so
  Resize+Conv is the sanctioned decoder path.
- `Gemm` alpha/beta/transpose — generalize `Linear` beyond the plain case.

## Tier 3 — bigger features (NOT started)

- `MatMul` + `LayerNormalization` / `RMSNormalization` — the transformer path
  (attention needs both + `Softmax`, which we have).
- `GRU` / `LSTM` — recurrent support, needs state handling across runs.
- Shape ops (`Transpose` / `Gather` / `Split` / `Slice` / `Pad`) — runtime has
  them, but shape juggling is where quantization/layout bugs hide; add on demand.
- Leftovers in registry, probably never needed: `Pow`, `Mod`, `ReduceL1/L2/Min/
  Max/Sum/Prod/Mean/SumSquare/LogSum/LogSumExp` (we cover `Mean`), `Gelu`+`LUT`
  (generic fallback), `QuantizeLinear`/`DequantizeLinear`/`RequantizeLinear`
  (PPQ plumbing), `SpaceToDepth`/`DepthToSpace`, `Squeeze`/`Unsqueeze`,
  `Reshape` (we rewrite to `Flatten`; runtime HAS it but old PPQ choked),
  `StreamingCache`/`InsertZeros`, `LpNormalization`, comparison ops
  (`Greater`, `Less`, `Equal`, ...), `ReverseSequence`, `ScatterND`, `Identity`,
  `LogSoftmax`, `PRelu`.

## Methodology lessons (from the Tier-1 investigation)

- FLOAT-input models (pure Log/Sqrt, PPQ keeps them F32) need FLOAT vectors in
the sketch; the old harness memcpy'd int8 into the float buffer (garbage.
log/sqrt numerics are covered by the composite instead, and `.scratch`
harnesses are git-ignored anyway), so fix the generator before relying on
standalone log/sqrt sweeps.

- Random-init nets are often UNQUANTIZABLE NOISE: 1 LSB of weight rounding can
  avalanche (e.g. through `Tanh`) into ±1.2 output swings. Host-side proof:
  float-chain-with-rounded-weights reproduces the device error almost exactly
  (1.19777 vs 1.20221 measured). Proof nets need trained-like magnitudes
  (conv weights ×0.15) — like trained weights have. Always verify the
  float-rounded gap is small BEFORE flashing.
- `sim == device` check: fixed-point host simulation from the `.info` tables
  (weights/bias/LUTs/scales) must match device RAW int8 to ±2 LSB. If it does,
  the device is exonerated; the gap is quantization proper.
- Sketch harness pitfalls: `.espdl` outputs are channel-last (NHWC) — refs must
  be transposed (`permute(0,2,3,1)`); `ExponentInfo.get(ch)` has NO bounds
  check — use `channel_size()` / per-tensor value, never guess the layout.
