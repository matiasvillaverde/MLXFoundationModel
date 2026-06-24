# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 22 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 33 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |
| `Sources/MLXLocalModels/Common/ModelConfiguration.swift` | Model identity, tokenizer overrides, local directory resolution, and generation token defaults. |
| `Sources/MLXLocalModels/Common/AbstractModelRegistry.swift` | Thread-safe model configuration lookup and fallback creation. |
| `Sources/MLXLocalModels/Common/ModelTypeRegistry.swift` | Thread-safe model type constructor lookup and unsupported-type reporting. |

## Recent Code Changes

Current independence pass:

- Replaced the shared model configuration and registry files listed above.
- Removed an unused prompt-preparation closure from `ModelConfiguration`; its only registry call site never affected runtime behavior because the closure was not stored or used.
- Added focused SwiftTesting coverage for remote and local model identity, equality, fallback configuration creation, replacement registration, constructor lookup, unsupported model errors, and concurrent registry writes.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-registry.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0659 | 0.0305 | 0.0355 | 225.38 | 121.31 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.0997 | 0.0367 | 0.0630 | 126.95 | 80.20 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.2002 | 0.0835 | 0.1167 | 68.56 | 39.95 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2908 | 0.1088 | 0.1820 | 43.95 | 27.51 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0901 | 0.0336 | 0.0565 | 141.59 | 88.81 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2706 | 0.1032 | 0.1675 | 47.77 | 29.56 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0453 | 0.0191 | 0.0263 | 304.55 | 176.45 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0720 | 0.0183 | 0.0537 | 130.46 | 97.25 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1470 | 0.0455 | 0.1015 | 78.79 | 54.43 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2963 | 0.1275 | 0.1688 | 47.39 | 27.00 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2517 | 0.1041 | 0.1476 | 54.21 | 31.79 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2377 | 0.0932 | 0.1445 | 55.36 | 33.65 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2223 | 0.0757 | 0.1466 | 54.57 | 35.99 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.9469 | 0.7381 | 0.2088 | 38.31 | 8.45 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0995 | 0.0372 | 0.0623 | 128.43 | 80.41 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1284 | 0.0376 | 0.0908 | 88.14 | 62.30 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1411 | 0.0540 | 0.0871 | 91.89 | 56.70 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1443 | 0.0566 | 0.0877 | 91.20 | 55.43 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1295 | 0.0540 | 0.0754 | 106.03 | 61.79 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3026 | 0.1187 | 0.1838 | 43.52 | 26.44 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2411 | 0.1947 | 0.0464 | 172.33 | 33.18 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2434 | 0.1966 | 0.0468 | 171.04 | 32.87 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 1.0241 | 0.9172 | 0.1068 | 74.88 | 7.81 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.6322 | 0.4459 | 0.1863 | 42.94 | 12.65 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.3082 | 0.1898 | 0.1184 | 67.57 | 25.96 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.3906 | 0.2583 | 0.1324 | 60.43 | 20.48 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1525 | 0.0644 | 0.0881 | 90.85 | 52.47 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0528 | 0.0203 | 0.0325 | 246.15 | 151.45 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4817 | 0.2035 | 0.2782 | 28.76 | 16.61 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0736 | 0.0177 | 0.0560 | 142.96 | 108.64 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.3287 | 0.1826 | 0.1460 | 54.78 | 24.34 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0954 | 0.0409 | 0.0545 | 146.76 | 83.83 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0660 | 0.0159 | 0.0501 | 159.78 | 121.29 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1136 | 0.0376 | 0.0759 | 105.36 | 70.44 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.8786 | 0.5626 | 0.3160 | 25.32 | 9.11 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2135 | 0.0739 | 0.1396 | 57.30 | 37.46 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3355 | 0.1490 | 0.1865 | 42.89 | 23.84 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2737 | 0.1011 | 0.1727 | 46.33 | 29.23 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3594 | 0.1374 | 0.2219 | 36.05 | 22.26 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1253 | 0.0477 | 0.0775 | 103.17 | 63.86 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0486 | 0.0116 | 0.0370 | 216.26 | 164.76 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2231 | 0.0800 | 0.1432 | 55.88 | 35.85 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0967 | 0.0470 | 0.0497 | 161.06 | 82.74 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2053 | 0.0913 | 0.1140 | 70.15 | 38.97 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4957 | 0.2238 | 0.2719 | 29.42 | 16.14 |
| `apertus` | `apertus` | 8 | 76 | 0.5493 | 0.3037 | 0.2456 | 32.57 | 14.56 |

## Skipped By Memory Gate

| Model | Reason |
| --- | --- |
| `qwen3-moe-30b-a3b-4bit` | Requires 48 GiB RAM. |
| `mistral-small-24b-2501-4bit` | Requires 48 GiB RAM. |
| `phi-3.5-moe-instruct-4bit` | Requires 48 GiB RAM. |
| `gemma-3n-e2b-it-lm-bf16` | Requires 48 GiB RAM. |
| `gemma-3n-e4b-it-lm-bf16` | Estimated runtime memory about 32 GiB; host budget is 24 GiB. |
| `deepseek-r1-4bit` | Requires 256 GiB RAM. |
| `cohere-command-r-v01-4bit` | Estimated runtime memory about 53 GiB; host budget is 24 GiB. |
| `gpt-oss` | Requires 48 GiB RAM. |
| `qwen3-next` | Requires 64 GiB RAM. |
| `qwen3.5-moe` | Requires 48 GiB RAM. |
