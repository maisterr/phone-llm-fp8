# 0 — Re-quantize with NVIDIA's selective recipe

**What.** Re-quantize via TensorRT Model Optimizer using NVIDIA's recipe for
the base model: keep self-attention + preceding Mamba layers in bf16,
quantize the KV cache.

**Why.** Fallback if the quality benchmark shows drift. Current recipe is
data-free and quantizes attention + Mamba projections (only `lm_head` + MoE
router stay bf16); NVIDIA's selective recipe is the proven ~99%-recovery
path for this architecture.

**Done when.** New checkpoint scores at parity in the quality benchmark AND
still fits one 48 GB L40S at `--max-model-len 32768`.

## Notes

- Blocked by: `0-fp8-quality-benchmark` — skip entirely if that shows no drift.
- Risk: larger bf16 share grows the checkpoint past 48 GB. Check size before
  touching the deploy; may force smaller max-model-len or kill the
  single-L40S premise.
