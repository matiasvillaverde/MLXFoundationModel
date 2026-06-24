# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 17 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 38 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |
| `Sources/MLXLocalModels/Common/ModelConfiguration.swift` | Model identity, tokenizer overrides, local directory resolution, and generation token defaults. |
| `Sources/MLXLocalModels/Common/ModelContainer.swift` | Actor-owned model context and prompt-cache access. |
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
- Replaced model container ownership with focused coverage for context updates, perform forwarding, legacy overload compatibility, and prompt-cache mutation.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-model-container.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0680 | 0.0323 | 0.0357 | 223.97 | 117.62 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1008 | 0.0386 | 0.0622 | 128.68 | 79.38 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1842 | 0.0671 | 0.1171 | 68.33 | 43.44 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2995 | 0.1174 | 0.1821 | 43.93 | 26.71 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0929 | 0.0361 | 0.0567 | 140.97 | 86.12 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2695 | 0.1022 | 0.1673 | 47.82 | 29.69 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0445 | 0.0182 | 0.0263 | 303.96 | 179.71 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0726 | 0.0181 | 0.0545 | 128.51 | 96.45 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1580 | 0.0560 | 0.1020 | 78.42 | 50.65 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3114 | 0.1430 | 0.1684 | 47.50 | 25.69 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2529 | 0.1047 | 0.1482 | 53.99 | 31.63 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2478 | 0.1045 | 0.1434 | 55.80 | 32.28 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2244 | 0.0782 | 0.1462 | 54.72 | 35.64 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.8281 | 0.6256 | 0.2026 | 39.49 | 9.66 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0890 | 0.0260 | 0.0630 | 126.92 | 89.85 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1334 | 0.0428 | 0.0906 | 88.31 | 59.95 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1550 | 0.0687 | 0.0864 | 92.61 | 51.60 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1296 | 0.0427 | 0.0869 | 92.11 | 61.74 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1289 | 0.0532 | 0.0757 | 105.68 | 62.05 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3039 | 0.1203 | 0.1837 | 43.56 | 26.32 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2373 | 0.1915 | 0.0458 | 174.58 | 33.71 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2392 | 0.1925 | 0.0467 | 171.38 | 33.45 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3943 | 0.2893 | 0.1050 | 76.18 | 20.29 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4893 | 0.3334 | 0.1559 | 51.30 | 16.35 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2021 | 0.1177 | 0.0843 | 94.86 | 39.59 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2904 | 0.1565 | 0.1340 | 59.72 | 27.55 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1446 | 0.0567 | 0.0878 | 91.07 | 55.33 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0452 | 0.0172 | 0.0280 | 285.51 | 176.80 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4501 | 0.1811 | 0.2690 | 29.74 | 17.77 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0783 | 0.0162 | 0.0621 | 128.85 | 102.22 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.2110 | 0.1033 | 0.1077 | 74.29 | 37.92 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0971 | 0.0429 | 0.0542 | 147.57 | 82.41 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0598 | 0.0149 | 0.0449 | 178.11 | 133.82 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1145 | 0.0377 | 0.0767 | 104.26 | 69.89 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6973 | 0.4094 | 0.2879 | 27.79 | 11.47 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2133 | 0.0740 | 0.1393 | 57.42 | 37.51 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2858 | 0.1009 | 0.1849 | 43.26 | 27.99 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2693 | 0.0963 | 0.1730 | 46.25 | 29.71 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3653 | 0.1428 | 0.2225 | 35.96 | 21.90 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1269 | 0.0492 | 0.0777 | 103.00 | 63.06 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0509 | 0.0108 | 0.0401 | 199.43 | 157.25 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2326 | 0.0885 | 0.1440 | 55.54 | 34.40 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0937 | 0.0436 | 0.0501 | 159.69 | 85.35 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2136 | 0.0985 | 0.1151 | 69.51 | 37.45 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4852 | 0.2156 | 0.2696 | 29.67 | 16.49 |
| `apertus` | `apertus` | 8 | 76 | 0.5559 | 0.3102 | 0.2456 | 32.57 | 14.39 |

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
