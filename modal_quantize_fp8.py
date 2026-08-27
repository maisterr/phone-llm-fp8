import modal


MODEL = "pipecat-ai/phonellm-alpha-1"
REVISION = "bd42ea2e560c34b5153d511e23a3f68727c724eb"
OUT_DIR = "/root/.cache/huggingface/checkpoints/phonellm-alpha-1-fp8-dynamic"

app = modal.App(name="phonellm-fp8-quantize")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "llmcompressor",
        "transformers",
        "accelerate",
        "huggingface-hub",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("phonellm-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    cpu=16,
    memory=131072,
    timeout=3 * 3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def quantize():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"Loading {MODEL}@{REVISION[:12]} in bf16 ...")
    # No trust_remote_code: the repo's custom code hard-requires mamba-ssm; the
    # native NemotronH class loads the same weights, and the "slow path" it
    # warns about never runs - data-free quantization does no forward passes.
    # device_map="cuda" (not "auto"): offloaded modules break save_pretrained.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype="auto",
        device_map="cuda",
    )
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    linear = [n for n, m in model.named_modules() if m.__class__.__name__ == "Linear"]
    print(f"{len(linear)} Linear modules; sample: {linear[:30]}")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        # lm_head and the MoE router stay bf16; extend (e.g. "re:.*mamba.*")
        # if output quality degrades on your domain.
        ignore=["lm_head", "re:.*router.*", "re:.*\\.gate$"],
    )
    oneshot(model=model, recipe=recipe)

    print(f"Saving compressed checkpoint to {OUT_DIR} ...")
    # save_original_format=False: transformers v5 cannot revert its fused-MoE
    # weight conversion at save time; vLLM accepts the converted layout.
    model.save_pretrained(
        OUT_DIR,
        save_compressed=True,
        max_shard_size="5GB",
        save_original_format=False,
    )
    tok.save_pretrained(OUT_DIR)
    hf_cache.commit()
    print("Done. Deploy the server with: modal deploy modal_l40s_fp8.py")


@app.local_entrypoint()
def main():
    quantize.remote()
