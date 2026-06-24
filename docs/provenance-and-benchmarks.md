# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 3 | Keep notices until replaced. |
| Explicit port or based-on markers | 41 | Keep source notes until replaced. |
| Files with neither marker | 51 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/MLXLLM/Gemma.swift` | Gemma RMSNorm, residual clipping, attention layout, decoder block, backbone, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Gemma2.swift` | Gemma2 soft-capped attention, grouped KV expansion, decoder block, backbone, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Internlm2.swift` | InternLM2 packed attention, dynamic RoPE planning, decoder blocks, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Phi3.swift` | Phi3 packed QKV attention, RoPE/LongRoPE planning, decoder block, backbone, tied/untied output heads, greedy-token fast path, config defaults, and LoRA target discovery. |
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
- Replaced Gemma with a shared project-owned norm, explicit attention layout, stable checkpoint keys, greedy-token fast path, and focused config/layout/LoRA coverage.
- Replaced Gemma2 with soft-capped attention layout, grouped KV expansion, greedy-token fast path, stable checkpoint keys, and focused config/layout/LoRA coverage.
- Replaced Phi3 with packed QKV layout, explicit RoPE/LongRoPE planning, tied/untied output handling, greedy-token fast path, and focused config/layout/LoRA coverage.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-phi3.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0637 | 0.0286 | 0.0351 | 228.18 | 125.58 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1024 | 0.0392 | 0.0632 | 126.55 | 78.12 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1915 | 0.0745 | 0.1169 | 68.41 | 41.78 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3012 | 0.1199 | 0.1814 | 44.11 | 26.56 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0943 | 0.0375 | 0.0568 | 140.86 | 84.85 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2704 | 0.1029 | 0.1675 | 47.77 | 29.59 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0452 | 0.0188 | 0.0264 | 303.34 | 176.93 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0732 | 0.0189 | 0.0543 | 128.83 | 95.63 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1486 | 0.0474 | 0.1011 | 79.10 | 53.85 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2991 | 0.1294 | 0.1697 | 47.14 | 26.75 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2529 | 0.1048 | 0.1482 | 54.00 | 31.63 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2316 | 0.0884 | 0.1432 | 55.87 | 34.55 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2286 | 0.0820 | 0.1467 | 54.54 | 34.99 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6817 | 0.4781 | 0.2036 | 39.29 | 11.73 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0904 | 0.0280 | 0.0624 | 128.25 | 88.50 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1360 | 0.0453 | 0.0907 | 88.22 | 58.81 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1501 | 0.0641 | 0.0860 | 92.98 | 53.29 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1282 | 0.0454 | 0.0828 | 96.63 | 62.42 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1204 | 0.0504 | 0.0700 | 114.25 | 66.43 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3045 | 0.1269 | 0.1776 | 45.04 | 26.27 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2367 | 0.1908 | 0.0459 | 174.28 | 33.80 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2492 | 0.2024 | 0.0468 | 170.97 | 32.10 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3785 | 0.2726 | 0.1059 | 75.56 | 21.14 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4830 | 0.3279 | 0.1551 | 51.57 | 16.56 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2114 | 0.1189 | 0.0925 | 86.47 | 37.84 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2839 | 0.1503 | 0.1336 | 59.86 | 28.18 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1463 | 0.0577 | 0.0886 | 90.27 | 54.69 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0455 | 0.0175 | 0.0280 | 285.49 | 175.79 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4469 | 0.1670 | 0.2799 | 28.58 | 17.90 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0751 | 0.0162 | 0.0589 | 135.90 | 106.56 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1803 | 0.0985 | 0.0818 | 97.83 | 44.38 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0991 | 0.0402 | 0.0589 | 135.80 | 80.69 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0617 | 0.0146 | 0.0471 | 170.00 | 129.74 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1117 | 0.0350 | 0.0768 | 104.23 | 71.60 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.7183 | 0.4215 | 0.2968 | 26.96 | 11.14 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2152 | 0.0752 | 0.1400 | 57.15 | 37.18 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3132 | 0.1258 | 0.1873 | 42.70 | 25.54 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2708 | 0.0984 | 0.1724 | 46.41 | 29.55 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.4014 | 0.1782 | 0.2232 | 35.85 | 19.93 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1261 | 0.0496 | 0.0765 | 104.52 | 63.44 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0477 | 0.0107 | 0.0370 | 216.29 | 167.71 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2330 | 0.0929 | 0.1401 | 57.10 | 34.34 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0929 | 0.0430 | 0.0499 | 160.40 | 86.12 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2216 | 0.1084 | 0.1131 | 70.72 | 36.11 |
| `olmo3` | `olmo3` | 8 | 85 | 0.5076 | 0.2372 | 0.2703 | 29.59 | 15.76 |
| `apertus` | `apertus` | 8 | 76 | 0.5641 | 0.3187 | 0.2454 | 32.60 | 14.18 |

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
