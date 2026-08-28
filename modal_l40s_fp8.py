import modal


MINUTES = 60
DEFAULT_PORT = 8000

# FP8 checkpoint produced by modal_quantize_fp8.py - run that once first.
MODEL_PATH = "/root/.cache/huggingface/checkpoints/phonellm-alpha-1-fp8-dynamic"
SERVED_MODEL_NAME = "pipecat-ai/phonellm-alpha-1"
REQUIRE_AUTHENTICATION = True
VLLM_VERSION = "0.27.1"
BASE_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu22.04"
AUTOINFERENCE_UTILS_VERSION = "0.2.6"

GPU_TYPE = "L40S"
N_GPUS = 1
GPU = f"{GPU_TYPE}:{N_GPUS}"
CPU = 16
MEMORY_MB = 65536

# 1 = one replica boots at deploy and stays warm (~$1.95/hr even idle);
# 0 = scale to zero, first request pays a multi-minute cold start.
MIN_CONTAINERS = 1

SCALEDOWN_WINDOW = 5 * MINUTES
TARGET_INPUTS = 40
STARTUP_TIMEOUT = 60 * MINUTES

HF_IMAGE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_OFFLINE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

vllm_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        f"autoinference-utils=={AUTOINFERENCE_UTILS_VERSION}",
        "httpx",
        "huggingface-hub",
    )
    .env(HF_IMAGE_ENV)
)

EXTRA_SERVER_ARGS = {
    "--async-scheduling": "",
    "--default-chat-template-kwargs": "{\"enable_thinking\":false}",
    "--enable-auto-tool-choice": "",
    "--enable-prefix-caching": "",
    "--gpu-memory-utilization": "0.90",
    "--mamba-cache-mode": "align",
    # Drop to 16384 if cache-block allocation fails at startup.
    "--max-model-len": "32768",
    "--override-generation-config": "{\"temperature\":0}",
    "--tool-call-parser": "qwen3_coder",
    # No --quantization flag: vLLM auto-detects the compressed-tensors FP8.
}

SERVER_ARGS = {
    "--served-model-name": SERVED_MODEL_NAME,
} | EXTRA_SERVER_ARGS


WARMUP_PAYLOAD = {
    "model": SERVED_MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with JSON facts about Tokyo."}],
    "max_tokens": 64,
    "temperature": 0,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "city_facts",
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "population": {"type": "integer"},
                },
                "required": ["city", "population"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
}


app = modal.App(name="ep-phonellm-alpha-1-l40s-fp8")


@app.cls(
    image=vllm_image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "phonellm-hf-cache", create_if_missing=True
        )
    },
)
@modal.concurrent(max_inputs=TARGET_INPUTS)
class Server:
    @modal.enter()
    def startup(self):
        from autoinference_utils.endpoint import VLLMEndpoint, warmup_chat_completions

        self.endpoint = VLLMEndpoint(
            model=MODEL_PATH,
            worker_port=DEFAULT_PORT,
            extra_server_args=SERVER_ARGS,
            health_timeout=STARTUP_TIMEOUT,
            health_poll_interval=5.0,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=DEFAULT_PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=2,
            request_timeout=60.0,
        )
        print(f"{SERVED_MODEL_NAME} ({GPU}, fp8) vllm deployment is ready.")

    @modal.web_server(
        DEFAULT_PORT,
        startup_timeout=STARTUP_TIMEOUT,
        requires_proxy_auth=REQUIRE_AUTHENTICATION,
    )
    def serve(self):
        # vLLM (started in @modal.enter) already listens on DEFAULT_PORT.
        pass

    @modal.exit()
    def stop(self):
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
