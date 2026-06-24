# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 12 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 43 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |
| `Sources/MLXLocalModels/Common/ModelConfiguration.swift` | Model identity, tokenizer overrides, local directory resolution, and generation token defaults. |
| `Sources/MLXLocalModels/Common/ModelContainer.swift` | Actor-owned model context and prompt-cache access. |
| `Sources/MLXLocalModels/Common/ModelFactory.swift` | Model context tokenization, factory dispatch, fallback errors, and trampoline registry. |
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
- Replaced model factory dispatch with focused coverage for chat-template tokenization, rendered/cache prompt encoding, factory fallback, final-error propagation, and missing-factory errors.
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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-model-factory.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0641 | 0.0290 | 0.0351 | 227.92 | 124.78 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1028 | 0.0397 | 0.0631 | 126.81 | 77.80 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1805 | 0.0642 | 0.1163 | 68.77 | 44.32 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3019 | 0.1204 | 0.1815 | 44.08 | 26.50 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0907 | 0.0341 | 0.0565 | 141.49 | 88.22 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2717 | 0.1056 | 0.1661 | 48.16 | 29.45 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0457 | 0.0185 | 0.0272 | 294.35 | 175.17 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0723 | 0.0187 | 0.0536 | 130.66 | 96.87 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1482 | 0.0475 | 0.1007 | 79.41 | 53.98 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2959 | 0.1270 | 0.1688 | 47.38 | 27.04 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2525 | 0.1043 | 0.1482 | 53.99 | 31.69 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2323 | 0.0880 | 0.1443 | 55.46 | 34.44 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2271 | 0.0814 | 0.1457 | 54.89 | 35.22 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.7247 | 0.5215 | 0.2032 | 39.38 | 11.04 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0907 | 0.0298 | 0.0609 | 131.40 | 88.22 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1282 | 0.0381 | 0.0901 | 88.75 | 62.39 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1485 | 0.0620 | 0.0864 | 92.56 | 53.89 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1509 | 0.0616 | 0.0893 | 89.56 | 53.01 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1263 | 0.0508 | 0.0755 | 105.99 | 63.33 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3117 | 0.1281 | 0.1836 | 43.58 | 25.67 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2410 | 0.1943 | 0.0467 | 171.38 | 33.20 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2431 | 0.1967 | 0.0464 | 172.29 | 32.91 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3820 | 0.2759 | 0.1061 | 75.42 | 20.94 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4685 | 0.3206 | 0.1480 | 54.07 | 17.07 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2278 | 0.1444 | 0.0834 | 95.91 | 35.11 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2970 | 0.1630 | 0.1340 | 59.71 | 26.94 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1459 | 0.0575 | 0.0884 | 90.49 | 54.85 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0459 | 0.0169 | 0.0290 | 275.92 | 174.39 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4493 | 0.1802 | 0.2691 | 29.73 | 17.81 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0782 | 0.0161 | 0.0621 | 128.91 | 102.37 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1669 | 0.0788 | 0.0881 | 90.84 | 47.93 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0934 | 0.0392 | 0.0542 | 147.53 | 85.61 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0635 | 0.0160 | 0.0475 | 168.60 | 126.04 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1124 | 0.0354 | 0.0769 | 104.01 | 71.21 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6910 | 0.4069 | 0.2841 | 28.15 | 11.58 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2136 | 0.0742 | 0.1394 | 57.40 | 37.46 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3273 | 0.1411 | 0.1862 | 42.97 | 24.44 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2779 | 0.1053 | 0.1727 | 46.33 | 28.78 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3809 | 0.1573 | 0.2236 | 35.78 | 21.00 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1310 | 0.0527 | 0.0784 | 102.08 | 61.05 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0533 | 0.0113 | 0.0419 | 190.89 | 150.22 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2387 | 0.0948 | 0.1440 | 55.57 | 33.51 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0931 | 0.0432 | 0.0498 | 160.51 | 85.94 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2092 | 0.0954 | 0.1138 | 70.29 | 38.24 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4914 | 0.2229 | 0.2684 | 29.80 | 16.28 |
| `apertus` | `apertus` | 8 | 76 | 0.5647 | 0.3172 | 0.2474 | 32.33 | 14.17 |

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
