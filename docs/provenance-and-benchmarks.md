# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 6 | Keep notices until replaced. |
| Explicit upstream model ports | 37 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 49 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/Common/Evaluate.swift` | Generation parameter normalization, sampler planning, processor planning, token iteration control, and generation result timing. |
| `Sources/MLXLocalModels/Common/GenerationConstants.swift` | Shared generation, cache, and sampling defaults. |
| `Sources/MLXLocalModels/Common/LanguageModel.swift` | Core model input/output contracts, default forwarding, greedy helpers, and cache creation. |
| `Sources/MLXLocalModels/Common/Load.swift` | Model artifact matching, deterministic safetensor discovery, and weight loading. |
| `Sources/MLXLocalModels/Common/LoRA+Layers.swift` | Dense and quantized LoRA replacement layers, adapter initialization, freeze policy, and fusion. |
| `Sources/MLXLocalModels/Common/Module+Extensions.swift` | Logical parameter counting for dense and quantized modules. |
| `Sources/MLXLocalModels/Common/RoPEApplication.swift` | Scalar and batch RoPE offset selection for KV caches. |
| `Sources/MLXLocalModels/Common/Tokenizer.swift` | Tokenizer loading, tokenizer-class rewriting, replacement registry, and streaming detokenization. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |
| `Sources/MLXLocalModels/MLXLLM/LLMModelFactory.swift` | LLM type registration, alias grouping, model load progress, generation-token resolution, and trampoline factory. |
| `Sources/MLXLocalModels/MLXLLM/Internlm2.swift` | InternLM2 packed attention, dynamic RoPE planning, decoder blocks, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Phi.swift` | Phi attention layout, decoder block, backbone, greedy-token fast path, configuration defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Lora+Data.swift` | LoRA JSONL/text data lookup and parsing. |
| `Sources/MLXLocalModels/MLXLLM/LoraTrain.swift` | LoRA batching, conversion/fusion, masked loss, evaluation, save/load, and training progress. |

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
- Replaced LoRA training helpers with focused coverage for shifted causal batches, prediction-length masking, weighted evaluation, adapter conversion/fusion, and quantized dequantize fusion.
- Replaced LLM model factory registration with grouped aliases, testable generation-token resolution, and focused coverage for alias registration plus EOS/suppress-token precedence.
- Replaced generation parameter and logit plan assembly with normalized inputs, explicit sampler/processor planning, and focused coverage for sampler selection plus active processor construction.
- Replaced LanguageModel core contracts with focused coverage for input slicing, media wrappers, default forwarding, greedy helpers, sanitize fallback, and KV-cache creation.
- Replaced model loading support with deterministic safetensor discovery, explicit directory errors, and focused coverage for recursive discovery, case-insensitive extensions, and missing directories.
- Replaced Phi with an explicit attention layout, project-owned module structure, config defaults, greedy-token fast path, and focused layout/config/LoRA coverage.
- Replaced InternLM2 with packed-attention layout, type-specific RoPE scaling, greedy-token fast path, packed LoRA targeting, and focused layout/RoPE/config coverage.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-internlm2.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0685 | 0.0325 | 0.0361 | 221.79 | 116.72 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1034 | 0.0392 | 0.0642 | 124.62 | 77.38 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.2015 | 0.0860 | 0.1155 | 69.26 | 39.71 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2939 | 0.1118 | 0.1822 | 43.92 | 27.22 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0903 | 0.0338 | 0.0566 | 141.46 | 88.58 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2929 | 0.1255 | 0.1674 | 47.78 | 27.31 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0501 | 0.0174 | 0.0327 | 244.64 | 159.55 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0741 | 0.0184 | 0.0557 | 125.69 | 94.51 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1548 | 0.0532 | 0.1016 | 78.71 | 51.67 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3024 | 0.1321 | 0.1702 | 47.00 | 26.46 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2468 | 0.0987 | 0.1480 | 54.04 | 32.42 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2297 | 0.0855 | 0.1442 | 55.46 | 34.82 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2250 | 0.0784 | 0.1466 | 54.58 | 35.55 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6345 | 0.4310 | 0.2035 | 39.31 | 12.61 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0905 | 0.0287 | 0.0618 | 129.47 | 88.42 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1273 | 0.0383 | 0.0889 | 89.95 | 62.85 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1413 | 0.0563 | 0.0850 | 94.09 | 56.60 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1410 | 0.0536 | 0.0874 | 91.54 | 56.72 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1241 | 0.0485 | 0.0756 | 105.83 | 64.47 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3128 | 0.1296 | 0.1832 | 43.66 | 25.57 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2411 | 0.1950 | 0.0461 | 173.56 | 33.18 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2417 | 0.1951 | 0.0466 | 171.68 | 33.10 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3937 | 0.2884 | 0.1054 | 75.92 | 20.32 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4672 | 0.3218 | 0.1454 | 55.01 | 17.12 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2098 | 0.1198 | 0.0899 | 88.95 | 38.14 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2830 | 0.1491 | 0.1339 | 59.75 | 28.27 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1471 | 0.0588 | 0.0883 | 90.59 | 54.39 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0451 | 0.0170 | 0.0281 | 284.96 | 177.31 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4624 | 0.1821 | 0.2803 | 28.54 | 17.30 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0736 | 0.0168 | 0.0568 | 140.77 | 108.68 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1602 | 0.0811 | 0.0791 | 101.12 | 49.93 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0942 | 0.0397 | 0.0545 | 146.83 | 84.94 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0581 | 0.0146 | 0.0435 | 184.06 | 137.80 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1128 | 0.0360 | 0.0767 | 104.25 | 70.94 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6670 | 0.3904 | 0.2766 | 28.92 | 11.99 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2144 | 0.0747 | 0.1396 | 57.29 | 37.32 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2936 | 0.1065 | 0.1870 | 42.78 | 27.25 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2770 | 0.1044 | 0.1726 | 46.35 | 28.88 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.4007 | 0.1778 | 0.2229 | 35.89 | 19.97 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1241 | 0.0482 | 0.0759 | 105.44 | 64.46 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0485 | 0.0108 | 0.0376 | 212.56 | 165.06 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2244 | 0.0832 | 0.1412 | 56.65 | 35.65 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0940 | 0.0444 | 0.0496 | 161.33 | 85.10 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2110 | 0.0969 | 0.1140 | 70.15 | 37.92 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4929 | 0.2224 | 0.2705 | 29.57 | 16.23 |
| `apertus` | `apertus` | 8 | 76 | 0.5580 | 0.3127 | 0.2453 | 32.61 | 14.34 |

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
