# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 13 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 42 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/Common/LanguageModel.swift` | Core model input/output contracts, default forwarding, greedy helpers, and cache creation. |
| `Sources/MLXLocalModels/Common/Load.swift` | Model artifact matching, deterministic safetensor discovery, and weight loading. |
| `Sources/MLXLocalModels/Common/Module+Extensions.swift` | Logical parameter counting for dense and quantized modules. |
| `Sources/MLXLocalModels/Common/RoPEApplication.swift` | Scalar and batch RoPE offset selection for KV caches. |
| `Sources/MLXLocalModels/Common/Tokenizer.swift` | Tokenizer loading, tokenizer-class rewriting, replacement registry, and streaming detokenization. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |
| `Sources/MLXLocalModels/MLXLLM/Lora+Data.swift` | LoRA JSONL/text data lookup and parsing. |

## Recent Code Changes

Current independence pass:

- Replaced the shared model configuration and registry files listed above.
- Removed an unused prompt-preparation closure from `ModelConfiguration`; its only registry call site never affected runtime behavior because the closure was not stored or used.
- Added focused SwiftTesting coverage for remote and local model identity, equality, fallback configuration creation, replacement registration, constructor lookup, unsupported model errors, and concurrent registry writes.
- Replaced default text-model prefill support with focused coverage for constants, default chunk size, explicit window chunking, and prompt-tail preservation.
- Replaced module parameter counting with a single-pass implementation and focused coverage for dense, embedding, and quantized module leaves.
- Replaced RoPE offset selection with focused coverage for nil-cache, scalar-cache, and per-row batch-cache paths.
- Replaced model container ownership with focused coverage for context updates, perform forwarding, legacy overload compatibility, and prompt-cache mutation.
- Replaced tokenizer support with focused coverage for tokenizer-class rewriting, registry updates, streaming deltas, newline resets, and incomplete Unicode boundaries.
- Replaced LoRA data loading with focused coverage for lookup precedence, JSONL parsing, text lines, missing files, and unsupported file types.
- Replaced LanguageModel core contracts with focused coverage for input slicing, media wrappers, default forwarding, greedy helpers, sanitize fallback, and KV-cache creation.
- Replaced model loading support with deterministic safetensor discovery, explicit directory errors, and focused coverage for recursive discovery, case-insensitive extensions, and missing directories.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-load-final.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0656 | 0.0305 | 0.0351 | 227.65 | 121.89 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1035 | 0.0401 | 0.0633 | 126.33 | 77.33 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.2131 | 0.0983 | 0.1148 | 69.71 | 37.54 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3010 | 0.1199 | 0.1811 | 44.18 | 26.58 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0917 | 0.0358 | 0.0559 | 143.13 | 87.24 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2752 | 0.1077 | 0.1675 | 47.76 | 29.07 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0455 | 0.0184 | 0.0271 | 295.59 | 175.84 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0740 | 0.0181 | 0.0560 | 125.11 | 94.54 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1587 | 0.0572 | 0.1015 | 78.81 | 50.42 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3053 | 0.1366 | 0.1687 | 47.41 | 26.20 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2529 | 0.1062 | 0.1467 | 54.53 | 31.64 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2328 | 0.0884 | 0.1444 | 55.39 | 34.37 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2302 | 0.0834 | 0.1467 | 54.52 | 34.76 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.8027 | 0.5999 | 0.2027 | 39.46 | 9.97 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0907 | 0.0286 | 0.0621 | 128.75 | 88.21 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1332 | 0.0425 | 0.0906 | 88.25 | 60.07 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1511 | 0.0649 | 0.0863 | 92.72 | 52.93 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1426 | 0.0551 | 0.0875 | 91.47 | 56.11 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1429 | 0.0663 | 0.0766 | 104.42 | 55.98 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3038 | 0.1205 | 0.1833 | 43.64 | 26.33 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2395 | 0.1928 | 0.0467 | 171.17 | 33.40 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2424 | 0.1967 | 0.0457 | 174.95 | 33.00 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3865 | 0.2811 | 0.1054 | 75.93 | 20.70 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4842 | 0.3333 | 0.1509 | 53.01 | 16.52 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2161 | 0.1319 | 0.0842 | 94.96 | 37.01 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2937 | 0.1604 | 0.1333 | 60.01 | 27.24 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1488 | 0.0601 | 0.0887 | 90.22 | 53.77 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0452 | 0.0171 | 0.0281 | 284.92 | 177.04 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4636 | 0.1932 | 0.2703 | 29.59 | 17.26 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0766 | 0.0158 | 0.0608 | 131.64 | 104.48 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1814 | 0.0929 | 0.0885 | 90.42 | 44.10 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0955 | 0.0410 | 0.0545 | 146.74 | 83.79 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0596 | 0.0146 | 0.0450 | 177.82 | 134.33 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1148 | 0.0388 | 0.0759 | 105.35 | 69.72 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6916 | 0.4167 | 0.2750 | 29.10 | 11.57 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2172 | 0.0774 | 0.1398 | 57.22 | 36.83 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3017 | 0.1155 | 0.1862 | 42.97 | 26.52 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2752 | 0.1042 | 0.1710 | 46.78 | 29.07 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3820 | 0.1587 | 0.2233 | 35.82 | 20.94 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1304 | 0.0521 | 0.0783 | 102.15 | 61.34 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0495 | 0.0103 | 0.0392 | 204.11 | 161.64 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2485 | 0.1048 | 0.1437 | 55.67 | 32.19 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0960 | 0.0457 | 0.0503 | 158.98 | 83.31 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2479 | 0.1343 | 0.1136 | 70.42 | 32.27 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4851 | 0.2148 | 0.2703 | 29.59 | 16.49 |
| `apertus` | `apertus` | 8 | 76 | 0.5598 | 0.3149 | 0.2449 | 32.66 | 14.29 |

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
