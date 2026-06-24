# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 8 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 47 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-evaluate.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0648 | 0.0298 | 0.0351 | 228.20 | 123.36 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1025 | 0.0390 | 0.0634 | 126.11 | 78.06 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1824 | 0.0656 | 0.1168 | 68.50 | 43.85 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3002 | 0.1181 | 0.1821 | 43.93 | 26.65 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0980 | 0.0409 | 0.0571 | 140.10 | 81.66 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2764 | 0.1086 | 0.1678 | 47.68 | 28.94 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0449 | 0.0186 | 0.0263 | 303.78 | 178.13 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0709 | 0.0185 | 0.0524 | 133.47 | 98.67 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1440 | 0.0431 | 0.1009 | 79.26 | 55.55 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3005 | 0.1318 | 0.1687 | 47.42 | 26.62 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2501 | 0.1020 | 0.1482 | 54.00 | 31.98 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2438 | 0.0996 | 0.1441 | 55.50 | 32.82 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2249 | 0.0792 | 0.1457 | 54.92 | 35.57 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.7439 | 0.5404 | 0.2035 | 39.32 | 10.75 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0942 | 0.0312 | 0.0630 | 126.94 | 84.95 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1308 | 0.0401 | 0.0907 | 88.20 | 61.18 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1641 | 0.0779 | 0.0862 | 92.84 | 48.75 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1318 | 0.0441 | 0.0877 | 91.24 | 60.72 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1520 | 0.0690 | 0.0830 | 96.44 | 52.65 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3135 | 0.1298 | 0.1838 | 43.54 | 25.52 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2482 | 0.2008 | 0.0474 | 168.75 | 32.23 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2398 | 0.1939 | 0.0459 | 174.17 | 33.36 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3801 | 0.2745 | 0.1056 | 75.75 | 21.05 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4695 | 0.3237 | 0.1458 | 54.87 | 17.04 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2238 | 0.1329 | 0.0909 | 87.97 | 35.74 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2814 | 0.1477 | 0.1337 | 59.81 | 28.43 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1503 | 0.0619 | 0.0883 | 90.55 | 53.24 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0473 | 0.0184 | 0.0289 | 276.54 | 168.96 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4377 | 0.1708 | 0.2669 | 29.97 | 18.28 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0768 | 0.0156 | 0.0611 | 130.88 | 104.21 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1953 | 0.0950 | 0.1003 | 79.76 | 40.97 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.1026 | 0.0463 | 0.0563 | 142.15 | 78.00 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0585 | 0.0147 | 0.0438 | 182.71 | 136.83 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1152 | 0.0384 | 0.0768 | 104.15 | 69.43 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6760 | 0.3877 | 0.2882 | 27.75 | 11.83 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2117 | 0.0708 | 0.1409 | 56.78 | 37.79 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2840 | 0.0966 | 0.1873 | 42.71 | 28.17 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2758 | 0.1042 | 0.1717 | 46.61 | 29.00 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3726 | 0.1495 | 0.2231 | 35.86 | 21.47 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1224 | 0.0448 | 0.0776 | 103.07 | 65.34 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0513 | 0.0116 | 0.0397 | 201.45 | 155.88 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2219 | 0.0780 | 0.1439 | 55.58 | 36.05 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0920 | 0.0430 | 0.0489 | 163.51 | 86.99 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2135 | 0.0982 | 0.1153 | 69.41 | 37.48 |
| `olmo3` | `olmo3` | 8 | 85 | 0.5029 | 0.2324 | 0.2705 | 29.57 | 15.91 |
| `apertus` | `apertus` | 8 | 76 | 0.5589 | 0.3139 | 0.2450 | 32.65 | 14.31 |

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
