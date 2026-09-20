# Detection roadmap — FOMO-style slim architectures

Goal: a working centroid detector (no bbox regression) that trains in `espdlx`
and runs fast on ESP32-S3, built on the Tier-2/3 ops (`Concat`, `Resize`, `Add`,
`MatMul` where useful). Status quo: `MobileNetV2.Fomo` reaches P=0.03/R=0.28 on
synthetic data only (`tests/test_fomo.py`) — the FOMO *idea* is unproven on real
data; everything below is ranked to fix that with the smallest arch first.

Reference failure analysis: the current Fomo feeds an 8-channel trunk straight
into a single 1×1 head at grid 12 — feature starvation at the head, no
multi-scale context, no spatial detail. The ranked archs inject detail via
Concat/Resize necks instead of widening the trunk.

## Ranking (build order)

1. **B — "Fomo-Slim"** (Concat + Resize top-down neck) — default arm.
2. **A — "Fomo-Nano"** (Concat two-tap, no Resize) — latency arm / ablation.
3. **C — "Fomo-MB"** (residual bottleneck backbone + B's neck) — accuracy arm.

B first: it is the smallest arch that carries the *whole* thesis (shallow
detail + deep semantics fused by the sanctioned `Resize→Conv→Concat` path, with
a grid-resolution switch that directly attacks the recall ceiling). A exists
mainly as the no-neck ablation that isolates B's contribution; C only makes
sense once B's head/loss design is proven on real data.

## Shared head contract (all archs)

- Output: `(N, C+1, G, G)` logits. Softmax at export/decode, never in the model
  (`decode_peaks` = 3×3 peak-NMS stays the decode).
- Single-class = C=1 → 2-logit `(bg, obj)` map, compatible with today's
  `fomo.py` helpers. Multi-class needs the C+1 loss extension below.
- BN before ReLU6 everywhere (folds at quantization); bounded activations keep
  the "trained-like magnitudes" contract from `OPS_ROADMAP.md`.
- Building block `DS(Cin→Cout, s)`: `DW3×3(s) + BN + ReLU6 → Conv1×1 + BN + ReLU6`.
- Structural rule (from `graph.py` threading): conv/resize run only on the
  current stream; saved tensors re-enter only via `Add` (same shape) or
  `Concat`. Lateral taps contribute their as-saved tensor — project them
  *before* the stream descends.

---

## Phase B — Fomo-Slim (top-down neck)

Table: | # | Layers | Out |
|---|---|---|
| 1 | Conv3×3 s2, 1→16, BN, ReLU6 | 48×48×16 |
| 2 | DS(16→24, s2) — **tap S saved** | 24×24×24 |
| 3 | DS(24→40, s2) | 12×12×40 |
| 4 | DS(40→40, s1) — deep | 12×12×40 |
| 5 | `Resize` nearest, ×2 (current = deep) | 24×24×40 |
| 6 | Conv1×1 40→24, BN, ReLU6 (top-down align) | 24×24×24 |
| 7 | `Concat[input_indices=[S, #6], dim=1]` | 24×24×48 |
| 8 | Conv3×3 48→32, BN, ReLU6 (fuse) | 24×24×32 |
| 9 | **B12:** Conv3×3 s2 32→32, BN, ReLU6 → head 1×1 @12 · **B24:** head 1×1 directly @24 | grid 12 / 24 |

~2.2 MMACs (B12) / ~2.7 MMACs (B24), ~35k params. New ops exercised:
`Concat`, `Resize` (nearest) — both Tier-2 device-proven.

- **B24 is the recall lever**: stride-4 grid on 96px = 576 cells (~4× grid-12
  granularity). FOMO recall is capped by centroids collapsing into one cell;
  the finer grid attacks that directly.
- B12 is the latency arm, same neck.
- Steps:
  1. Zoo entry `Fomo.Slim(grid=12|24)` + shape validation at 96px (and a
     128px variant if a dataset needs it — arch is size-agnostic above 96).
  2. Multiclass loss helper in `espdlx.fomo`: C+1 weighted CE for positives,
     `(1-heat)^gamma`-shaped negatives for the background channel; keep
     `init_bias_from_prior` (generalize bias init to the argmax class).
  3. Synthetic tests (reuse `tests/test_fomo.py` pattern), then ONE real
     dataset end-to-end: train → `convert()` → device latency + MATCH check on
     the head output (spatially sensitive — do NOT end the proof chain in
     Mean+Gemm; compare per-cell logits vs host, see Tier-2 spatial caveat).
  4. Tuning sweep (only after a working baseline): grid 12 vs 24,
     `object_weight`, `sigma`/`gamma` of `soft_loss`, nearest vs linear Resize
     at step 5 (linear is device-proven too, costs more).
- Exit criteria: real-data P/R clearly above the current Fomo baseline
  (target: R ≥ 0.6 at P ≥ 0.5 to call the head design validated), device
  latency measured, `.espdl` MATCH on per-cell outputs.
- Risks: Concat quantization (per-tensor scales on merged branches — keep both
  inputs post-ReLU6 so ranges are bounded and similar); Resize ×2 nearest
  carries 2×2 block artifacts (acceptable; linear is the fallback).

## Phase A — Fomo-Nano (two-tap concat)

Table: | # | Layers | Out |
|---|---|---|
| 1 | Conv3×3 s2, 1→16, BN, ReLU6 | 48×48×16 |
| 2 | DS(16→24, s2) — tap M saved | 24×24×24 |
| 3 | DS(24→32, s2) — **tap D saved** | 12×12×32 |
| 4 | DS(32→48, s1) | 12×12×48 |
| 5 | Conv3×3 48→48, BN, ReLU6 | 12×12×48 |
| 6 | `Concat[input_indices=[D, cur], dim=1]` | 12×12×80 |
| 7 | Conv3×3 80→40, BN, ReLU6 (fuse) | 12×12×40 |
| 8 | Conv1×1 40→C+1 (head) | 12×12×C+1 |

~1.9 MMACs, ~25k params. Grid 12 only. No Resize: the "neck" is just a
deep-tap concat, so it doubles as the ablation proving (or refuting) that B's
top-down path — not merely feature concat — is what buys recall.

- Steps: same shared head work as B (build AFTER B so the loss helpers exist);
  train + benchmark on the same dataset split as B for a fair A-vs-B table.
- Exit criteria: latency table entry (expected fastest), accuracy delta vs B
  recorded. If A ≈ B, drop the Resize from the production arch and keep A.
- Risks: lowest, but also lowest ceiling — expected to trail B on small
  objects.

## Phase C — Fomo-MB (residual bottleneck backbone)

Same neck as B (Resize → 1×1 align → Concat → fuse → head), backbone swapped
for MBConv blocks: `1×1 expand → DW3×3 → 1×1 project`, residual via
`Add(input_index=i)` (in==out channels, same shape — Tier-1-proven op, but the
zoo's residual-free habit was a latency choice: measure the skip overhead).

Table: | # | Layers | Out |
|---|---|---|
| 1 | Conv3×3 s2, 3→24 (RGB), BN, ReLU6 | 48×48×24 |
| 2 | 1 MBConv(24, exp48, s1) @48 | 48×48×24 |
| 3 | MBConv s2 → 32, then 1 MBConv(32, exp96, s1) — **tap S saved** | 24×24×32 |
| 4 | MBConv(32→40, exp96, s1) ×2 | 24×24×40 |
| 5 | MBConv s2 → 64 | 12×12×64 |
| 6 | MBConv(64, exp128, s1) — deep | 12×12×64 |
| 7 | `Resize` ×2 → 1×1 64→32 → `Concat[tap S, up]` → fuse Conv3×3 64→32 | 24×24×32 |
| 8 | Head 1×1 32→C+1 @24 (or s2 → head @12) | grid 24 / 12 |

~5–6 MMACs, ~90k params. RGB input assumed (3-channel stem).

- Gate: build ONLY if B (and A) plateau on real data. It reuses B's
  head/loss/decode stack verbatim — no new training machinery, just backbone
  capacity.
- Exit criteria: accuracy above B by a meaningful margin to justify the ~2×
  MACs; otherwise keep B and archive C.
- Risks: residual Add on device adds a scale-align pass per block (latency);
  large-magnitude spatials must stay inside the trained-like contract.

---

## Milestone summary

| Milestone | Arch | Gate |
|---|---|---|
| M1 | B zoo entry + C+1 loss helpers | synthetic tests green, real-data baseline |
| M2 | B on-device end-to-end | latency + per-cell MATCH vs host |
| M3 | B24 recall report vs B12 | R target ≥ 0.6 @ P ≥ 0.5 |
| M4 | A benchmark arm | fair A-vs-B table on the same split |
| M5 | C (only if gated in) | beats B by enough to pay ~2× MACs |
| M6 | Multi-class real run | C+1 head, peak-NMS decode on device |

Deliberately out of scope (why): multi-head multi-scale outputs (single-output
`Model`), `ConvTranspose` decoders (not in the registry), bbox regression
(FOMO premise), `SpaceToDepth`/`DepthToSpace` tricks (not in the registry).

---

## Progress log (cat5k runs, all @160px grid 40, soft_loss, peak-NMS k5 eval)

| run | arch | obj_w | epochs | val F1 best | note |
|---|---|---|---|---|---|
| old fomo160a–e | trunk v0 (no neck) | 20–100 | 25–100 | ~0.18 | P≈0.1 R≈0.1–0.5 — reference floor |
| fomoslim_b | B width 1.0 gray | 25 | 150 | 0.278 | FPs: 70% far-background |
| fomoslim_b50 | B width 1.0 gray | 50 | 150 | 0.278 (P 0.28/R 0.28 @0.7) | exports clean |
| fomoslim_b (obj 100) | B width 1.0 gray | 100 | 150 | 0.256 | R 0.57 but P 0.13 — too hot |
| fomoslim_b400 | B width 1.0 gray | 50 | 400 | **0.298** (P 0.24/R 0.39 @0.5) | weights +40% vs b50 → PPQ exponent assert on the deep DW conv — NOT deployable without weight decay |
| fomoslim_w15rgb | B width 1.5 RGB | 50 | 400 | 0.311 (@t0.7) | capacity+color did NOT break the plateau — objective, not capacity |
| fomoslim_cp | B + copy-paste aug, wd 1e-3 | 50 | 400 (killed ~ep245) | 0.276 | aug plateau — small-cat recall unchanged |
| fomoslim_i224 | B @224px grid 56 | 50 | 400 (killed ~ep200) | 0.271 | resolution did not break the plateau |
| fomoslim_hm / hm075 | soft→heatmap regression | 50 | 400 (killed ~ep165/205) | 0.176 / 0.206 | worse: with σ=1.0 the target bump keeps ring cells at h≈0.6 > decode thr → model TRAINS to fire multi-cell plateaus; σ=0.75/r1 still short of soft_loss |
| memorization probe | B width 1.0, no aug, 512 imgs | 50 | 120 | train F1 0.78 (P .85/R .72) | ARCH CAN LEARN THE TASK — it's generalization, not capacity |
| fomoslim_ceiling best-by-val | same | 50 | 120 | val 0.204 | best-by-val-F1 selects early snapshots; final weights fit train much better |

Findings so far:
- Arch B beats every prior FOMO attempt on the same data (0.298 vs ~0.18),
  but is below the roadmap exit target (R ≥ 0.6 @ P ≥ 0.5).
- obj_w ~50 is the F1 sweet spot; 100 trades P for R at fixed F1.
- 5×5 peak-NMS (k5) slightly beats 3×3 at stride-4 grids (F1 +0.01).
- FP forensics @k5 t0.6: 70% far-background, 18% localization jitter,
  12% duplicate peaks in one GT — an underfitting problem, hence width+RGB.
- Size-band forensics: recall small(<0.1) 0.275 / mid 0.45 / large 0.755 —
  the misses are SMALL cats (161/234 FNs). Large cats detect fine.
- Memorization diagnostic: no-aug 512-img run reaches train F1 0.78 → the
  loss/arch can express the task; the wall is generalization (3.3k images).
- Width 1.5 + RGB, copy-paste aug, 224px resolution: all plateau ≈ same F1.
- heatmap (CenterNet-style) regression: WORSE on this task — ring cells at
  h>0.5 decode as extra peaks (plateau firing), plus shape supervision is
  stiffer than hard-cell supervision here.
- DEPLOYMENT BLOCKER found: long no-decay schedules inflate conv weights
  (~40%) and violate esp-dl's int8 exponent budget (out_exp ≥ in0_exp +
  w_exp). Mitigation: AdamW weight decay (keeps weights trained-like).
  All future runs MUST set `--wd` (1e-3) or the `.espdl` export aborts.
- CONCLUSION: with only ~3.4k COCO-cat images the FOMO-per-cell ceiling on
  this val is ≈ F1 0.30. Remaining lever: multi-class COCO pretrain
  (118k images, 81-ch head) → backbone transfer to the cat head
  (`.scratch/fomo_coco_train.py` + `modal_coco_full.py`).
