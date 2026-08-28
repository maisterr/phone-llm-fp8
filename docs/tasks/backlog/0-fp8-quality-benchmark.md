# 0 — Benchmark FP8 quality vs bf16

**What.** Real eval of the FP8 checkpoint against the bf16 endpoint on
voice-agent scenarios: tool calls, strict-JSON output, multi-turn dialogue.

**Why.** Current evidence is spot checks only — byte-identical at
`temperature=0` on a handful of prompts. Encouraging, not an eval. README
claims quality parity it cannot back yet.

**Done when.** Score table FP8 vs bf16 on real scenarios lands in README,
replacing the "not yet properly benchmarked" caveat. Gap per task <1% or
documented.

## Notes

- Reference point: NVIDIA's official FP8 of the base model
  (Nemotron 3 Nano 30B-A3B) reports ~99% median accuracy recovery.
  Expected outcome; this fine-tune + our data-free recipe still need proof.
- Reuse `modal_bench.py` harness for request plumbing where possible.
