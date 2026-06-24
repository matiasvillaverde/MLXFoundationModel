# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 20 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 35 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |
| `Sources/MLXLocalModels/Common/ModelConfiguration.swift` | Model identity, tokenizer overrides, local directory resolution, and generation token defaults. |
| `Sources/MLXLocalModels/Common/AbstractModelRegistry.swift` | Thread-safe model configuration lookup and fallback creation. |
| `Sources/MLXLocalModels/Common/ModelTypeRegistry.swift` | Thread-safe model type constructor lookup and unsupported-type reporting. |
| `Sources/MLXLocalModels/Common/GenerationConstants.swift` | Shared generation, cache, and sampling defaults. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |

## Recent Code Changes

Current independence pass:

- Replaced the shared model configuration and registry files listed above.
- Removed an unused prompt-preparation closure from `ModelConfiguration`; its only registry call site never affected runtime behavior because the closure was not stored or used.
- Added focused SwiftTesting coverage for remote and local model identity, equality, fallback configuration creation, replacement registration, constructor lookup, unsupported model errors, and concurrent registry writes.
- Replaced default text-model prefill support with focused coverage for constants, default chunk size, explicit window chunking, and prompt-tail preservation.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-prefill.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0693 | 0.0330 | 0.0363 | 220.59 | 115.49 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1016 | 0.0383 | 0.0633 | 126.43 | 78.73 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1917 | 0.0750 | 0.1166 | 68.58 | 41.74 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3017 | 0.1204 | 0.1813 | 44.12 | 26.52 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0893 | 0.0328 | 0.0565 | 141.62 | 89.60 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2761 | 0.1087 | 0.1674 | 47.78 | 28.97 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0487 | 0.0214 | 0.0273 | 292.75 | 164.14 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0710 | 0.0181 | 0.0528 | 132.54 | 98.66 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1460 | 0.0457 | 0.1003 | 79.79 | 54.80 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2953 | 0.1263 | 0.1690 | 47.32 | 27.09 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2465 | 0.0990 | 0.1476 | 54.21 | 32.45 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2291 | 0.0847 | 0.1445 | 55.37 | 34.91 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2253 | 0.0788 | 0.1465 | 54.60 | 35.51 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.8802 | 0.6752 | 0.2049 | 39.04 | 9.09 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0919 | 0.0307 | 0.0612 | 130.81 | 87.08 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1286 | 0.0379 | 0.0907 | 88.22 | 62.22 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1804 | 0.0938 | 0.0866 | 92.38 | 44.35 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1351 | 0.0438 | 0.0912 | 87.67 | 59.22 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1322 | 0.0574 | 0.0749 | 106.84 | 60.49 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3107 | 0.1268 | 0.1840 | 43.49 | 25.75 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2402 | 0.1939 | 0.0463 | 172.63 | 33.30 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2482 | 0.2014 | 0.0468 | 170.83 | 32.23 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.4234 | 0.3045 | 0.1189 | 67.28 | 18.90 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4750 | 0.3201 | 0.1550 | 51.63 | 16.84 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2078 | 0.1182 | 0.0897 | 89.23 | 38.49 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2923 | 0.1591 | 0.1332 | 60.07 | 27.37 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1483 | 0.0599 | 0.0884 | 90.48 | 53.93 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0500 | 0.0190 | 0.0310 | 257.72 | 159.90 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4488 | 0.1822 | 0.2665 | 30.01 | 17.83 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0771 | 0.0161 | 0.0610 | 131.14 | 103.71 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1840 | 0.0928 | 0.0912 | 87.76 | 43.49 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0976 | 0.0400 | 0.0576 | 138.98 | 81.99 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0607 | 0.0148 | 0.0458 | 174.50 | 131.89 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1113 | 0.0345 | 0.0768 | 104.17 | 71.90 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6782 | 0.3845 | 0.2938 | 27.23 | 11.80 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2147 | 0.0748 | 0.1399 | 57.19 | 37.26 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2915 | 0.1072 | 0.1843 | 43.42 | 27.45 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2787 | 0.1060 | 0.1727 | 46.31 | 28.70 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3736 | 0.1513 | 0.2223 | 35.99 | 21.41 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1310 | 0.0529 | 0.0782 | 102.34 | 61.05 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0516 | 0.0107 | 0.0410 | 195.32 | 154.95 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2148 | 0.0706 | 0.1442 | 55.49 | 37.25 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0897 | 0.0429 | 0.0468 | 170.94 | 89.18 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2127 | 0.0989 | 0.1138 | 70.29 | 37.61 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4914 | 0.2212 | 0.2703 | 29.60 | 16.28 |
| `apertus` | `apertus` | 8 | 76 | 0.5675 | 0.3211 | 0.2465 | 32.46 | 14.10 |

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
