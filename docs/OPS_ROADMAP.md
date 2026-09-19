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

## Tier 2 — SHIPPED (device-proven 2026-09-19, chain level)

- `Concat` — multi-input `Model` support shipped: `input_indices=[...]`
  (list of saved layer outputs, order = concat order), `dim` arg. Unlocks
  DenseNet / FPN-style nets. Export emits a single `Concat(axis)` node
  (ESP-PPQ converts axis NCHW->NHWC: composite `.espdl` carries `axis: 3`).
- `Resize` — nearest / linear / bicubic (ONNX `Resize`; runtime
  `dl::Resize` NEAREST/LINEAR/CUBIC) via `scale_factor` or `size`, with
  `coordinate_transformation_mode` plumbing (`half_pixel` / `asymmetric` /
  `align_corners`). Note: **no `ConvTranspose` in the registry**, so
  Resize+Conv is the sanctioned decoder path. `size=` (4-input ONNX
  Resize) breaks ESP-PPQ's calibration executor, so `make_espdl_friendly`
  rewrites sizes->scales (`sizes / input_shape`, exact for integer pixels).
- `Gemm` — generalizes `Linear` with ONNX semantics
  `alpha * A·B + beta * C`, `transB` (weight stored as the ONNX `B`;
  `transA` rejected — runtime asserts 0). Plain
  `alpha=beta=1, transB=True` reuses the tier-1-proven `nn.Linear` path
  (`Gemm(transB=1)`); any other combo emits `alpha`/`beta` attrs which
  `convert.make_espdl_friendly` folds into the weight/bias initializers
  (ESP-PPQ asserts `alpha == beta == 1.0`), so the device still runs a
  plain `dl::Gemm` (composite `.espdl` shows `alpha:1 beta:1 transB:0`).

Host proof (`.scratch/tier2_proof.py`, Mul-treatment steps 1-3):
- unit parity + shapes: 57 tests green (tests/test_layers.py,
  tests/test_model.py, tests/test_convert.py).
- ONNX export emits `Concat`/`Resize`/`Gemm` nodes, `onnx.checker`-clean
  (Resize-containing models export at opset 18 — the torch version converter
  has no 17->13 adapter for `Resize`; real ESP-PPQ accepts opset 18).
- REAL `espdl_quantize_onnx` (no stubs) produces composite `.espdl` from
  one chain: dense block (3-input Concat) -> Resize nearest -> Conv -> Resize
  linear -> Conv -> Mean -> Flatten -> Gemm; fbs inspection shows
  `Concat(axis=3)`, `Resize(mode='n'/'l', coordinate='a'/'h')`, `Gemm(1/1/0)`,
  plus a standalone alpha/beta Gemm model folded to a plain `dl::Gemm`.

Device run (steps 4-5, 2026-09-19, ESP32-S3 @ 240 MHz): 3/3 MATCH
(maxabs 0.011/0.012/0.017 vs the 0.3 sketch bound; host sim predicted
0.013). Concat/Resize/Gemm resolve and run; both Resize modes
(nearest + bilinear) measured in-chain. Also fixed en route: the proof
script's sketch referenced a wrong header var (`tier2_proof_espdl` vs the
generated `tier2_composite_espdl`) — model renamed to `tier2_proof` so
`convert()` outputs match the sketch; future skew fails at compile time.

Honest caveat (read before claiming spatial fidelity): the composite ends
in Mean+Gemm, so the MATCH proves chain-level correctness (ops resolve,
scales and channel-means right) but is insensitive to spatial noise — the
Mean washes it out. Raw spatial conv outputs measured separately show
magnitude-dependent fixed-point noise (bit-exact on small signals,
deviating on large ones; e.g. mean ~5 LSB @exp-12 for 1x1 x0.15, worse at
Kaiming scale). Ruled out as causes: NHWC layout, exponents, LUT scales
(`check_luts.py` clean), shifts, permutations, replay infidelity
(same-graph sims), padding, ReLU fusion, library regression (Tier-1
reproduces at 0.003 on the same toolchain). Constraint (not defect): proof
and production nets must use trained-like magnitudes (the Tier-1 x0.15
rule) — Kaiming-scale spatials are out of contract. Root-causing the conv
kernel's large-signal behavior is a separate esp-nn audit, not blocking:
all shipped proofs use trained-like magnitudes and match.

## Tier 3 — SHIPPED (device-proven 2026-09-19)

Transformer path + shape ops in one composite chain (`.scratch/tier3_proof.py`,
full Mul treatment steps 1-5). Sketch at `Arduino/projects/TF/tier3_proof`
(external/git-ignored). 3/3 MATCH on ESP32-S3 @ 240 MHz (maxabs 0.096/0.092/
0.092 against the 0.1 bound; out int8 exp -7, in exp -5).

Device-verdict anatomy (the honest version of step 5):

- `sim == device` is bit-exact (0.0 LSB, all 3 vecs, replayed EXACT convert
  incl. calib): every Tier-3 op (`MatMul`, `LayerNormalization`, fused
  `RMSNormalization`, `Transpose`, `Reshape`, `Slice`, `Gather`, `Pad`,
  `Squeeze`/`Unsqueeze`, `Softmax`) computes correctly on-device. The
  residual float gap is quantization peak-compression, not a device defect.
- Error shape: 26/32 outputs sub-2-LSB (median sub-LSB); outliers sit only
  on the largest-magnitude elements. An attention chain (2 softmaxes +
  3 matmuls + 2 norms, int8) does not hit Tier-1's 0.003 — 0.1 max / 0.013
  mean is the measured floor for this proof net (Tier-2 accepted 0.3).
- Run-to-run `.espdl` bytes differ (PYTHONHASHSEED content-hash IDs) while
  all scales/zero-points are identical multisets — serialization noise,
  quantization itself deterministic and reproducible.

- `MatMul` — binary `forward(x, y)`, skip (`input_index=i`: current @ saved)
  or saved pair (`input_indices=[i, j]`, order = operand order): `Q @ K^T`
  and `weights @ V` without leaving the `Model`. No transpose flags — compose
  with `Transpose` (keeps export 1:1). Rank >= 2, inner dims match, batch
  broadcasts (esp-dl `MatMul` requirement, fail fast).
- `LayerNorm` — wraps `nn.LayerNorm`, exports natively as
  `LayerNormalization` (opset 17+; Tier-3 models export at opset 18 — same
  precedent as Tier-2 `Resize`).
- `RMSNorm` — `weight * (x / sqrt(mean(pow(x,2)) + eps))` over trailing dims.
  The `torch.pow` (not `x*x`) and negative reduction dims are load-bearing:
  ESP-PPQ's importer fuses exactly `Pow -> ReduceMean -> [Add] -> Sqrt ->
  Div -> Mul` with negative axes into one native `RMSNormalization`
  (`FORMATTER_FUSE_RMSNORM`); positive axes abort the fuse (no static rank
  at import) and `Mul` stays six quantized kernels. Composite `.espdl` shows
  a single `RMSNormalization(axis=-1)`. Identity scales (init ones) are
  folded by onnxsim before PPQ ever sees them — proof nets use trained-like
  scales (uniform 0.8–1.2), same methodology as conv x0.15.
- Shape ops: `Transpose(perm)` (full permutation, negatives ok), `Reshape`
  (**per-sample** shape — batch prepended as leading `-1`, so calibration
  batch-N and device batch-1 agree; a baked batch breaks calibration),
  `Squeeze`/`Unsqueeze`, `Slice` (ONNX clamp semantics, positive steps),
  `Gather` (frozen index list via `index_select`; no `input_index` mode — a
  float stream cannot supply int64 indices), `Pad` (constant only).
  `Split` returns a tuple (manual wiring; exports native `Split` at opset
  18) — inside a `Model` it fails fast pointing at `Slice` decomposition
  (same runtime op; `Model` guards multi-output layers in both `forward`
  and `validate_shapes`). Head split/merge stays single-tensor via
  `Reshape`+`Transpose` (no Split needed for attention).
- `convert.make_espdl_friendly` changes: `Reshape` rewrites to `Flatten`
  ONLY when flatten-equivalent (genuine head reshapes preserved for
  `dl::Reshape` — the old blind rewrite would corrupt them); a leading dim
  folded to the static batch (exporter artifact when the input comes from a
  `Transpose`) is restored to `-1` (identical at batch-1, correct at
  calibration batch-N).

Host proof (`.scratch/tier3_proof.py`): unit parity + shapes (79 tests green);
ONNX export checker-clean at opset 18 (`MatMul`, `LayerNormalization`,
`Pow`-headed norm composite, `Transpose`/`Reshape`/`Slice`/`Gather`/`Pad`/
`Squeeze`/`Unsqueeze`, standalone `Split`); REAL `espdl_quantize_onnx` (no
stubs) on the 20-layer composite — fbs shows `MatMul`, `Transpose`,
`LayerNormalization`, fused `RMSNormalization`, `Reshape(allowzero=1)`,
`Slice`, `Gather`, `Pad`, `Unsqueeze`, `Squeeze`; ORT(float) vs torch 1e-6.

Debugging history (how the 3/3 was earned — read before writing proof nets):

- First flash DIFFed with byte-identical outputs across vecs: the proof net
  was input-independent IN FLOAT (two stacked x0.15 Linears crushed the
  signal 10x per layer, then Softmax saturated). Caught by per-layer
  `vecdiff` tracing, not by parity tests (a degenerate net passes parity).
  Fix: Linears keep Kaiming scale (already trained-like for projections).
- Second flash gave constant-per-vec outputs with float refs also constant:
  raw `Q@K^T` scores without Softmax amplify (~0.2 -> ~5.3 peaks) past int8
  range. Fix: Softmax after head scores (the realistic pattern) bounds the
  chain to ±0.95 and all vecs vary.

Deferred with reason:

- `GRU` / `LSTM` — recurrent support needs state handling across runs, but
  `Model` is a stateless feedforward stack. Stateful recurrence is a separate
  design (streaming state / `StreamingCache`), not another layer class.
- Leftovers in registry, probably never needed: `Mod`, `ReduceL1/L2/Min/
  Max/Sum/Prod/SumSquare/LogSum/LogSumExp` (we cover `Mean`), `Gelu`+`LUT`
  (generic fallback), `SpaceToDepth`/`DepthToSpace`, `StreamingCache`/
  `InsertZeros`, `LpNormalization`, comparison ops, `ReverseSequence`,
  `ScatterND`, `Identity`, `LogSoftmax`, `PRelu`. (`Pow`, `ReduceMean`,
  `QuantizeLinear`/`DequantizeLinear`, `Squeeze`/`Unsqueeze`, `Reshape` are
  now covered via Tier 3; `QuantizeLinear`/`DequantizeLinear` remain PPQ
  plumbing, not user layers.)

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
