# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 7 | Keep notices until replaced. |
| Explicit upstream model ports | 38 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 48 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-phi.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0690 | 0.0328 | 0.0362 | 220.93 | 115.91 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1027 | 0.0397 | 0.0630 | 126.94 | 77.89 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.2114 | 0.0947 | 0.1167 | 68.56 | 37.85 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3057 | 0.1238 | 0.1820 | 43.97 | 26.17 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0923 | 0.0357 | 0.0566 | 141.26 | 86.67 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2610 | 0.0938 | 0.1673 | 47.83 | 30.65 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0463 | 0.0189 | 0.0274 | 291.99 | 172.66 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0735 | 0.0188 | 0.0547 | 128.02 | 95.27 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1548 | 0.0532 | 0.1016 | 78.74 | 51.67 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2970 | 0.1282 | 0.1688 | 47.38 | 26.94 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2510 | 0.1030 | 0.1480 | 54.06 | 31.87 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2247 | 0.0806 | 0.1442 | 55.49 | 35.60 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2186 | 0.0721 | 0.1464 | 54.63 | 36.60 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6245 | 0.4209 | 0.2036 | 39.29 | 12.81 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0907 | 0.0291 | 0.0616 | 129.88 | 88.16 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1315 | 0.0421 | 0.0894 | 89.52 | 60.85 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1683 | 0.0821 | 0.0863 | 92.74 | 47.53 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1292 | 0.0425 | 0.0866 | 92.34 | 61.94 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1268 | 0.0437 | 0.0830 | 96.35 | 63.11 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3094 | 0.1257 | 0.1837 | 43.56 | 25.86 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2392 | 0.1932 | 0.0460 | 173.98 | 33.45 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2418 | 0.1959 | 0.0459 | 174.25 | 33.08 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3731 | 0.2675 | 0.1056 | 75.77 | 21.44 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4821 | 0.3267 | 0.1554 | 51.47 | 16.59 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2123 | 0.1278 | 0.0845 | 94.63 | 37.68 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2843 | 0.1508 | 0.1335 | 59.92 | 28.14 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1500 | 0.0623 | 0.0877 | 91.20 | 53.32 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0466 | 0.0171 | 0.0295 | 271.32 | 171.56 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4716 | 0.1914 | 0.2801 | 28.56 | 16.96 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0747 | 0.0157 | 0.0590 | 135.69 | 107.12 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1665 | 0.0779 | 0.0887 | 90.20 | 48.03 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0951 | 0.0402 | 0.0549 | 145.85 | 84.13 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0599 | 0.0148 | 0.0450 | 177.67 | 133.61 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1150 | 0.0384 | 0.0766 | 104.48 | 69.57 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6441 | 0.3641 | 0.2800 | 28.57 | 12.42 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2178 | 0.0779 | 0.1399 | 57.19 | 36.74 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3046 | 0.1174 | 0.1872 | 42.74 | 26.26 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2750 | 0.1021 | 0.1729 | 46.27 | 29.09 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3842 | 0.1609 | 0.2233 | 35.83 | 20.82 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1260 | 0.0484 | 0.0776 | 103.06 | 63.47 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0524 | 0.0111 | 0.0413 | 193.49 | 152.62 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2242 | 0.0803 | 0.1439 | 55.58 | 35.68 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0927 | 0.0433 | 0.0493 | 162.12 | 86.34 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2150 | 0.1009 | 0.1141 | 70.13 | 37.21 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4903 | 0.2194 | 0.2710 | 29.52 | 16.31 |
| `apertus` | `apertus` | 8 | 76 | 0.5631 | 0.3188 | 0.2443 | 32.75 | 14.21 |

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
