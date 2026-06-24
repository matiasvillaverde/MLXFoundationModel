# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 4 | Keep notices until replaced. |
| Explicit port or based-on markers | 41 | Keep source notes until replaced. |
| Files with neither marker | 50 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-gemma2.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0642 | 0.0290 | 0.0352 | 227.59 | 124.68 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1031 | 0.0398 | 0.0633 | 126.45 | 77.63 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1807 | 0.0638 | 0.1170 | 68.39 | 44.26 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2960 | 0.1155 | 0.1806 | 44.31 | 27.03 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0955 | 0.0370 | 0.0585 | 136.68 | 83.76 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2697 | 0.1023 | 0.1674 | 47.78 | 29.66 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0463 | 0.0189 | 0.0274 | 291.47 | 172.71 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0745 | 0.0182 | 0.0563 | 124.36 | 93.92 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1545 | 0.0532 | 0.1013 | 78.99 | 51.79 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2914 | 0.1228 | 0.1687 | 47.43 | 27.45 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2622 | 0.1141 | 0.1481 | 54.01 | 30.51 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2305 | 0.0865 | 0.1440 | 55.55 | 34.70 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2260 | 0.0775 | 0.1486 | 53.85 | 35.40 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.7675 | 0.5633 | 0.2043 | 39.16 | 10.42 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0902 | 0.0273 | 0.0629 | 127.29 | 88.70 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1321 | 0.0417 | 0.0904 | 88.49 | 60.55 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1486 | 0.0623 | 0.0863 | 92.70 | 53.83 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1338 | 0.0517 | 0.0821 | 97.47 | 59.80 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1265 | 0.0550 | 0.0715 | 111.90 | 63.23 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3080 | 0.1309 | 0.1772 | 45.16 | 25.97 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2361 | 0.1901 | 0.0460 | 173.98 | 33.88 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2398 | 0.1927 | 0.0471 | 169.86 | 33.36 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3841 | 0.2676 | 0.1165 | 68.68 | 20.83 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4837 | 0.3291 | 0.1547 | 51.72 | 16.54 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2134 | 0.1140 | 0.0994 | 80.47 | 37.49 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2833 | 0.1498 | 0.1335 | 59.93 | 28.24 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1503 | 0.0618 | 0.0885 | 90.37 | 53.22 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0460 | 0.0174 | 0.0286 | 279.43 | 173.82 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4698 | 0.1924 | 0.2774 | 28.84 | 17.03 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0742 | 0.0156 | 0.0586 | 136.55 | 107.77 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1512 | 0.0851 | 0.0660 | 121.14 | 52.92 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0971 | 0.0407 | 0.0564 | 141.85 | 82.35 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0599 | 0.0147 | 0.0452 | 177.11 | 133.57 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1162 | 0.0405 | 0.0757 | 105.72 | 68.87 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6849 | 0.4035 | 0.2814 | 28.43 | 11.68 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2144 | 0.0748 | 0.1395 | 57.34 | 37.32 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2885 | 0.1010 | 0.1875 | 42.67 | 27.73 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2737 | 0.1008 | 0.1729 | 46.28 | 29.23 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3714 | 0.1479 | 0.2235 | 35.79 | 21.54 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1204 | 0.0427 | 0.0777 | 103.02 | 66.46 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0490 | 0.0104 | 0.0386 | 207.22 | 163.13 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2244 | 0.0832 | 0.1413 | 56.64 | 35.65 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0953 | 0.0436 | 0.0517 | 154.86 | 83.95 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2219 | 0.1060 | 0.1159 | 69.01 | 36.05 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4930 | 0.2248 | 0.2682 | 29.83 | 16.23 |
| `apertus` | `apertus` | 8 | 76 | 0.5891 | 0.3448 | 0.2443 | 32.75 | 13.58 |

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
