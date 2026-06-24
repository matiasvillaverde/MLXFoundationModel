# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 18 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 37 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/Common/RoPEApplication.swift` | Scalar and batch RoPE offset selection for KV caches. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |

## Recent Code Changes

Current independence pass:

- Replaced the shared model configuration and registry files listed above.
- Removed an unused prompt-preparation closure from `ModelConfiguration`; its only registry call site never affected runtime behavior because the closure was not stored or used.
- Added focused SwiftTesting coverage for remote and local model identity, equality, fallback configuration creation, replacement registration, constructor lookup, unsupported model errors, and concurrent registry writes.
- Replaced default text-model prefill support with focused coverage for constants, default chunk size, explicit window chunking, and prompt-tail preservation.
- Replaced module parameter counting with a single-pass implementation and focused coverage for dense, embedding, and quantized module leaves.
- Replaced RoPE offset selection with focused coverage for nil-cache, scalar-cache, and per-row batch-cache paths.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-rope-offset.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0665 | 0.0309 | 0.0355 | 225.16 | 120.38 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1023 | 0.0393 | 0.0630 | 126.89 | 78.18 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1849 | 0.0680 | 0.1169 | 68.44 | 43.27 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3016 | 0.1197 | 0.1820 | 43.97 | 26.52 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0934 | 0.0366 | 0.0568 | 140.82 | 85.62 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2708 | 0.1037 | 0.1671 | 47.86 | 29.54 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0447 | 0.0187 | 0.0260 | 307.46 | 178.91 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0719 | 0.0182 | 0.0537 | 130.30 | 97.35 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1526 | 0.0511 | 0.1015 | 78.82 | 52.44 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3057 | 0.1372 | 0.1685 | 47.48 | 26.17 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2513 | 0.1037 | 0.1476 | 54.19 | 31.83 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2357 | 0.0914 | 0.1443 | 55.43 | 33.94 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2278 | 0.0820 | 0.1459 | 54.84 | 35.11 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.7963 | 0.5904 | 0.2058 | 38.86 | 10.05 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0926 | 0.0315 | 0.0611 | 131.02 | 86.42 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1312 | 0.0413 | 0.0899 | 88.95 | 60.97 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1662 | 0.0797 | 0.0865 | 92.47 | 48.14 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1814 | 0.0896 | 0.0918 | 87.12 | 44.09 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1446 | 0.0700 | 0.0746 | 107.27 | 55.32 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3054 | 0.1217 | 0.1837 | 43.56 | 26.20 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2446 | 0.1986 | 0.0460 | 174.01 | 32.71 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2425 | 0.1958 | 0.0467 | 171.30 | 33.00 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3815 | 0.2761 | 0.1054 | 75.91 | 20.97 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4658 | 0.3180 | 0.1478 | 54.14 | 17.17 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2152 | 0.1242 | 0.0910 | 87.93 | 37.18 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2798 | 0.1484 | 0.1314 | 60.90 | 28.59 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1454 | 0.0570 | 0.0884 | 90.54 | 55.02 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0453 | 0.0172 | 0.0281 | 284.77 | 176.70 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4458 | 0.1777 | 0.2680 | 29.85 | 17.95 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0732 | 0.0159 | 0.0573 | 139.53 | 109.27 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1828 | 0.0902 | 0.0926 | 86.38 | 43.77 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0958 | 0.0396 | 0.0562 | 142.38 | 83.48 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0589 | 0.0146 | 0.0444 | 180.32 | 135.78 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1122 | 0.0355 | 0.0767 | 104.33 | 71.33 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.7078 | 0.4178 | 0.2900 | 27.58 | 11.30 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2157 | 0.0758 | 0.1400 | 57.16 | 37.09 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2840 | 0.0993 | 0.1847 | 43.32 | 28.17 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2832 | 0.1100 | 0.1732 | 46.19 | 28.25 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.4052 | 0.1825 | 0.2227 | 35.93 | 19.74 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1206 | 0.0436 | 0.0770 | 103.93 | 66.34 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0465 | 0.0105 | 0.0361 | 221.91 | 171.94 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2253 | 0.0818 | 0.1435 | 55.75 | 35.50 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0944 | 0.0447 | 0.0496 | 161.20 | 84.79 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2062 | 0.0924 | 0.1138 | 70.33 | 38.81 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4796 | 0.2091 | 0.2705 | 29.57 | 16.68 |
| `apertus` | `apertus` | 8 | 76 | 0.5540 | 0.3103 | 0.2437 | 32.82 | 14.44 |

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
