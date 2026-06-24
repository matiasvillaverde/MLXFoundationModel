# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 11 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 44 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/Common/LoRA+Layers.swift` | Dense and quantized LoRA replacement layers, adapter initialization, freeze policy, and fusion. |
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
- Replaced LoRA layer adapters with focused coverage for dense/quantized conversion, adapter-only training, no-op initialization, fusion, and quantized mode preservation.
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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-lora-layers.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0715 | 0.0347 | 0.0368 | 217.42 | 111.94 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1002 | 0.0371 | 0.0632 | 126.66 | 79.81 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1799 | 0.0641 | 0.1158 | 69.08 | 44.48 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2961 | 0.1158 | 0.1802 | 44.38 | 27.02 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0920 | 0.0356 | 0.0564 | 141.75 | 86.93 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2708 | 0.1044 | 0.1663 | 48.10 | 29.55 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0459 | 0.0180 | 0.0279 | 286.76 | 174.20 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0720 | 0.0188 | 0.0532 | 131.57 | 97.27 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1466 | 0.0451 | 0.1015 | 78.79 | 54.57 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2922 | 0.1233 | 0.1690 | 47.35 | 27.38 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2493 | 0.1013 | 0.1480 | 54.05 | 32.09 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2127 | 0.0683 | 0.1444 | 55.41 | 37.61 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2197 | 0.0730 | 0.1468 | 54.50 | 36.41 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6783 | 0.4748 | 0.2035 | 39.31 | 11.79 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0893 | 0.0275 | 0.0618 | 129.45 | 89.61 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1289 | 0.0385 | 0.0905 | 88.43 | 62.05 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1871 | 0.1009 | 0.0863 | 92.75 | 42.75 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1284 | 0.0368 | 0.0916 | 87.32 | 62.32 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1344 | 0.0593 | 0.0751 | 106.48 | 59.51 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3105 | 0.1270 | 0.1835 | 43.59 | 25.77 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2409 | 0.1949 | 0.0460 | 173.80 | 33.20 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2393 | 0.1934 | 0.0460 | 173.99 | 33.43 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3898 | 0.2746 | 0.1152 | 69.42 | 20.52 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4751 | 0.3273 | 0.1478 | 54.11 | 16.84 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.1984 | 0.1144 | 0.0840 | 95.24 | 40.33 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2853 | 0.1517 | 0.1336 | 59.87 | 28.04 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1453 | 0.0574 | 0.0878 | 91.09 | 55.07 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0446 | 0.0171 | 0.0275 | 290.94 | 179.20 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4362 | 0.1669 | 0.2692 | 29.71 | 18.34 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0772 | 0.0158 | 0.0614 | 130.38 | 103.68 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.2041 | 0.1035 | 0.1006 | 79.52 | 39.20 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0961 | 0.0425 | 0.0536 | 149.23 | 83.22 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0593 | 0.0146 | 0.0446 | 179.23 | 135.01 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1151 | 0.0386 | 0.0766 | 104.46 | 69.47 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.7780 | 0.4703 | 0.3078 | 25.99 | 10.28 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2184 | 0.0785 | 0.1400 | 57.16 | 36.63 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2845 | 0.0975 | 0.1870 | 42.77 | 28.12 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2749 | 0.1025 | 0.1724 | 46.41 | 29.10 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3793 | 0.1560 | 0.2233 | 35.82 | 21.09 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1229 | 0.0452 | 0.0777 | 102.90 | 65.08 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0508 | 0.0107 | 0.0400 | 199.76 | 157.57 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2340 | 0.0903 | 0.1437 | 55.69 | 34.19 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.1061 | 0.0481 | 0.0580 | 138.03 | 75.41 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2139 | 0.0984 | 0.1156 | 69.23 | 37.39 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4937 | 0.2247 | 0.2690 | 29.74 | 16.20 |
| `apertus` | `apertus` | 8 | 76 | 0.5687 | 0.3224 | 0.2463 | 32.47 | 14.07 |

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
