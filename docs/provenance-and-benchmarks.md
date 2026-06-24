# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 19 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 36 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |
| `Sources/MLXLocalModels/Common/ModelConfiguration.swift` | Model identity, tokenizer overrides, local directory resolution, and generation token defaults. |
| `Sources/MLXLocalModels/Common/AbstractModelRegistry.swift` | Thread-safe model configuration lookup and fallback creation. |
| `Sources/MLXLocalModels/Common/ModelTypeRegistry.swift` | Thread-safe model type constructor lookup and unsupported-type reporting. |
| `Sources/MLXLocalModels/Common/GenerationConstants.swift` | Shared generation, cache, and sampling defaults. |
| `Sources/MLXLocalModels/Common/Module+Extensions.swift` | Logical parameter counting for dense and quantized modules. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |

## Recent Code Changes

Current independence pass:

- Replaced the shared model configuration and registry files listed above.
- Removed an unused prompt-preparation closure from `ModelConfiguration`; its only registry call site never affected runtime behavior because the closure was not stored or used.
- Added focused SwiftTesting coverage for remote and local model identity, equality, fallback configuration creation, replacement registration, constructor lookup, unsupported model errors, and concurrent registry writes.
- Replaced default text-model prefill support with focused coverage for constants, default chunk size, explicit window chunking, and prompt-tail preservation.
- Replaced module parameter counting with a single-pass implementation and focused coverage for dense, embedding, and quantized module leaves.

Previous performance pass:

`Sources/MLXLocalModels/Common/AttentionUtils.swift` now separates the normal
attention path from the shared-KV path. When a caller only needs attention
output, quantized KV caches no longer materialize dequantized key and value
arrays that are immediately discarded.

This should help when runtime KV quantization is enabled. The real-model sweep
below is mainly a regression check for normal model execution.

## Environment

| Field | Value |
| --- | --- |
| Mac | MacBook Pro, Mac14,5 |
| Chip | Apple M2 Max |
| CPU cores | 12: 8 performance, 4 efficiency |
| Unified memory | 32 GB |
| macOS | 27.0, build 26A5353q |
| Xcode | 27.0, build 27A5194q |
| Swift | 6.4 |
| Model storage | `.build/test-models` |

Command:

```sh
MLX_TEST_MODELS_DIR="$PWD/.build/test-models" \
MLX_HOST_MEMORY_GB=32 \
MLX_REAL_MODEL_GENERATION_TOKENS=8 \
MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS=240 \
MLX_REAL_MODEL_TIMEOUT_SECONDS=1200 \
MLX_REAL_MODEL_FEATURE_TIMEOUT_SECONDS=900 \
CONFIGURATION=release \
make test-all-architectures
```

## E2E Result

The all-architecture sweep passed for every model selected by the memory gate.
The test runner selected 46 downloadable models and skipped 10 oversized models
on this 32 GB host. Each selected model ran serialized generation, rendered
session requests, and token-level grammar constraint checks.

## Benchmarks

These rows come from `BENCH` lines printed by the real-model test runner in
`.build/benchmarks/test-all-architectures-2026-06-24-independent-module-count.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0623 | 0.0279 | 0.0344 | 232.37 | 128.42 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.0984 | 0.0351 | 0.0633 | 126.34 | 81.27 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1914 | 0.0764 | 0.1150 | 69.54 | 41.80 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3000 | 0.1197 | 0.1804 | 44.35 | 26.66 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0913 | 0.0348 | 0.0565 | 141.48 | 87.60 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2693 | 0.1028 | 0.1665 | 48.06 | 29.71 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0456 | 0.0182 | 0.0274 | 291.68 | 175.28 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0697 | 0.0185 | 0.0511 | 136.98 | 100.50 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1464 | 0.0459 | 0.1005 | 79.62 | 54.64 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3136 | 0.1446 | 0.1690 | 47.33 | 25.51 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2461 | 0.0989 | 0.1472 | 54.34 | 32.50 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2319 | 0.0877 | 0.1442 | 55.47 | 34.50 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2229 | 0.0763 | 0.1466 | 54.56 | 35.89 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.7637 | 0.5606 | 0.2031 | 39.40 | 10.48 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0930 | 0.0302 | 0.0628 | 127.33 | 85.99 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1344 | 0.0439 | 0.0904 | 88.45 | 59.55 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1610 | 0.0748 | 0.0863 | 92.72 | 49.68 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1338 | 0.0450 | 0.0888 | 90.11 | 59.79 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1360 | 0.0611 | 0.0749 | 106.80 | 58.81 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3017 | 0.1190 | 0.1827 | 43.79 | 26.52 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2405 | 0.1942 | 0.0463 | 172.72 | 33.27 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2376 | 0.1916 | 0.0460 | 173.91 | 33.68 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3750 | 0.2693 | 0.1057 | 75.72 | 21.33 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4672 | 0.3192 | 0.1480 | 54.06 | 17.12 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2138 | 0.1242 | 0.0897 | 89.23 | 37.41 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2842 | 0.1503 | 0.1338 | 59.77 | 28.15 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1462 | 0.0575 | 0.0887 | 90.18 | 54.71 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0464 | 0.0172 | 0.0292 | 274.08 | 172.42 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4719 | 0.1920 | 0.2799 | 28.58 | 16.95 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0765 | 0.0156 | 0.0608 | 131.48 | 104.62 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1856 | 0.0960 | 0.0896 | 89.24 | 43.10 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0950 | 0.0410 | 0.0539 | 148.31 | 84.24 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0601 | 0.0158 | 0.0443 | 180.76 | 133.15 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1104 | 0.0343 | 0.0761 | 105.11 | 72.44 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6899 | 0.3979 | 0.2920 | 27.40 | 11.60 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2189 | 0.0800 | 0.1388 | 57.62 | 36.55 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3112 | 0.1262 | 0.1849 | 43.26 | 25.71 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2733 | 0.1014 | 0.1720 | 46.52 | 29.27 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3580 | 0.1361 | 0.2219 | 36.06 | 22.35 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1320 | 0.0533 | 0.0787 | 101.65 | 60.61 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0482 | 0.0104 | 0.0378 | 211.64 | 166.03 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2148 | 0.0708 | 0.1439 | 55.58 | 37.25 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0950 | 0.0460 | 0.0490 | 163.20 | 84.23 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2195 | 0.1058 | 0.1137 | 70.36 | 36.45 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4914 | 0.2238 | 0.2677 | 29.89 | 16.28 |
| `apertus` | `apertus` | 8 | 76 | 0.5506 | 0.3038 | 0.2468 | 32.41 | 14.53 |

## Skipped By Memory Gate

| Model | Reason | Present in `.build/test-models` |
| --- | --- | --- |
| `qwen3-moe-30b-a3b-4bit` | Requires 48 GiB RAM. | No. |
| `mistral-small-24b-2501-4bit` | Requires 48 GiB RAM. | No. |
| `phi-3.5-moe-instruct-4bit` | Requires 48 GiB RAM. | No. |
| `gemma-3n-e2b-it-lm-bf16` | Requires 48 GiB RAM. | No. |
| `gemma-3n-e4b-it-lm-bf16` | Estimated runtime memory about 32 GiB; host budget is 24 GiB. | Yes, `gemma-3n-E4B-it-lm-bf16`. |
| `deepseek-r1-4bit` | Requires 256 GiB RAM. | No. |
| `cohere-command-r-v01-4bit` | Estimated runtime memory about 53 GiB; host budget is 24 GiB. | Yes, `c4ai-command-r-v01-4bit`. |
| `gpt-oss` | Requires 48 GiB RAM. | No. |
| `qwen3-next` | Requires 64 GiB RAM. | No. |
| `qwen3.5-moe` | Requires 48 GiB RAM. | No. |
