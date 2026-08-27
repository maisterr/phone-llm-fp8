import asyncio
import json
import os
import statistics
import time

import modal

app = modal.App(name="phonellm-bench")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("aiohttp")

# Resolved only inside the container - the local modal CLI env has no aiohttp.
with image.imports():
    import aiohttp

SYSTEM = (
    "You are a voice assistant for a service business, speaking to a customer "
    "on the phone. Speak briefly and naturally. Your job is to schedule "
    "appointments: find out what the customer needs, then a preferred day and "
    "time. Business hours are Monday-Saturday, 9:00 to 19:00. One reply is "
    "one or two sentences. No lists or formatting - this is spoken "
    "conversation."
)

USER_TURNS = [
    "Hi, I'd like to book an appointment for next week.",
    "Hello, something stopped working and I need someone to take a look - when can I come in?",
    "Hi, do you have anything available tomorrow morning?",
    "Good afternoon, how much does a standard check-up cost?",
    "Hello, I'd like to book a repair for Saturday.",
    "Hi, I need a routine replacement done - what days are free?",
    "Good afternoon, are you open on Sundays?",
    "Hello, I have an urgent issue - can I come in today?",
]


@app.function(image=image, cpu=4, timeout=30 * 60)
async def bench(
    base_url: str,
    model: str,
    headers: dict,
    levels: list,
    ttft_limit_ms: float,
    min_tps: float,
    max_tokens: int,
    rounds: int,
) -> None:
    async def one_call(session, idx):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                # Unique suffix defeats response caching; the shared system
                # prompt still hits the prefix cache, like real calls do.
                {"role": "user", "content": f"{USER_TURNS[idx % len(USER_TURNS)]} (call #{idx})"},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.perf_counter()
        ttft = None
        n_tokens = 0
        usage_completion = None
        try:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as r:
                if r.status != 200:
                    return {"error": f"HTTP {r.status}: {(await r.text())[:120]}"}
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage_completion = chunk["usage"].get("completion_tokens")
                    for choice in chunk.get("choices", []):
                        if choice.get("delta", {}).get("content"):
                            if ttft is None:
                                ttft = time.perf_counter() - start
                            n_tokens += 1
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        total = time.perf_counter() - start
        if ttft is None:
            return {"error": "no content received"}
        out = usage_completion or n_tokens
        tps = (out - 1) / (total - ttft) if total > ttft and out > 1 else 0.0
        return {"ttft": ttft, "total": total, "tokens": out, "tps": tps}

    def pct(values, p):
        values = sorted(values)
        k = max(0, min(len(values) - 1, round(p / 100 * (len(values) - 1))))
        return values[k]

    async with aiohttp.ClientSession() as session:
        print(f"Endpoint: {base_url}  model: {model}  (load generated inside Modal)")
        print(f"Usable = ttft-p95 <= {ttft_limit_ms:.0f} ms AND every stream >= {min_tps:.0f} tok/s")
        warm = await one_call(session, 0)
        if "error" in warm:
            print(f"Warmup failed: {warm['error']} - is the endpoint up?")
            return
        print(f"Warm: ttft {warm['ttft']*1000:.0f} ms, {warm['tps']:.1f} tok/s")
        print(f"Rounds per level: {rounds} (stats pooled across rounds)\n")
        print("conc | ttft-p50 | ttft-p95 | tok/s-p50 | tok/s-min | agg-tok/s | errs | verdict")
        outcomes = []
        for n in levels:
            ok, errors, walls = [], [], []
            for rnd in range(rounds):
                t0 = time.perf_counter()
                results = await asyncio.gather(
                    *(one_call(session, rnd * n + i) for i in range(n))
                )
                walls.append(time.perf_counter() - t0)
                ok += [r for r in results if "error" not in r]
                errors += [r["error"] for r in results if "error" in r]
                await asyncio.sleep(2)
            if not ok:
                print(f"{n:>4} | ALL FAILED: {errors[:2]}")
                continue
            ttfts = [r["ttft"] for r in ok]
            tpss = [r["tps"] for r in ok]
            ttft_p95 = pct(ttfts, 95)
            usable = ttft_p95 * 1000 <= ttft_limit_ms and min(tpss) >= min_tps and not errors
            outcomes.append((n, usable))
            print(
                f"{n:>4} | {statistics.median(ttfts)*1000:7.0f} | {ttft_p95*1000:7.0f} | "
                f"{statistics.median(tpss):8.1f} | {min(tpss):8.1f} | "
                f"{sum(r['tokens'] for r in ok) / sum(walls):8.1f} | {len(errors):>4} | "
                f"{'OK' if usable else 'over'}"
            )
        good = [n for n, u in outcomes if u]
        print()
        if good:
            print(f"Usable concurrency (measured from inside Modal): up to ~{max(good)} calls.")
        else:
            print("No tested level met the threshold.")


# Minimal .env parser - python-dotenv may be absent in the modal CLI env.
def _read_env(path: str) -> dict:
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return vals


@app.local_entrypoint()
def main(
    levels: str = "1,2,4,8,16,32",
    ttft_ms: float = 500.0,
    min_tps: float = 15.0,
    max_tokens: int = 80,
    rounds: int = 3,
):
    env = {**_read_env(os.path.join(os.path.dirname(__file__), ".env")), **os.environ}
    base = env["MODAL_LLM_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    key = env.get("MODAL_LLM_API_KEY", "")
    if env.get("MODAL_KEY") and env.get("MODAL_SECRET"):
        headers = {"Modal-Key": env["MODAL_KEY"], "Modal-Secret": env["MODAL_SECRET"]}
    elif key.startswith("wk-") and ".ws-" in key:
        wk, ws = key.split(".ws-", 1)
        headers = {"Modal-Key": wk, "Modal-Secret": f"ws-{ws}"}
    else:
        headers = {"Authorization": f"Bearer {key or 'EMPTY'}"}
    bench.remote(
        base_url=base,
        model=env["MODAL_LLM_MODEL"],
        headers=headers,
        levels=[int(x) for x in levels.split(",")],
        ttft_limit_ms=ttft_ms,
        min_tps=min_tps,
        max_tokens=max_tokens,
        rounds=rounds,
    )
