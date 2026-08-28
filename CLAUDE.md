# This Project

Serves `pipecat-ai/phonellm-alpha-1` (30B Mamba-Transformer MoE, bf16-only)
on one Modal L40S by quantizing it once to an offline FP8 checkpoint.
OpenAI-compatible endpoint for voice agents.

- `modal_quantize_fp8.py` — one-time FP8 quantization (H200), writes checkpoint to `phonellm-hf-cache` Volume. `modal run`.
- `modal_l40s_fp8.py` — vLLM server, 1x L40S. `modal deploy`, never `modal run` (no entrypoint).
- `modal_bench.py` — TTFT/throughput benchmark. `modal run`, needs `.env` (see `.env.example`).
- `docs/tasks/` — work board, folder = status. Read `docs/tasks/README.md`.

Every GPU command costs real money. `MIN_CONTAINERS = 1` keeps an L40S warm
24/7 (~$1.95/hr). Confirm before deploy, quantize, or config that boots GPUs.

## Style

BLUF: bottom line first line. Answer, then evidence, then detail on request.

Speak compressed caveman. Keep all technical substance. Kill only fluff.

- Drop articles, filler, pleasantries, hedging, praise.
- Pattern: [thing] [action] [reason]. [next step].
- Short synonyms: big not extensive, fix not "implement a solution for". Technical terms exact.
- Code blocks unchanged. Quote errors exact. Fragments fine.
- One fact one line. Line gone, fact gone? No = delete.
- Applies to chat, docs, comments, commits.

## Rules

- Evidence before claims. Read source first.
- Never enter unmeasured claim into README (this repo sells numbers).
- No time estimates for engineering effort.
- Minimal must-have tests only. In doubt, no test.
