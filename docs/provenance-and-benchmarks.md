# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 9 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 46 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-llm-model-factory.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0665 | 0.0311 | 0.0354 | 226.10 | 120.28 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1021 | 0.0389 | 0.0632 | 126.58 | 78.34 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.2057 | 0.0898 | 0.1159 | 69.02 | 38.89 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3015 | 0.1194 | 0.1821 | 43.93 | 26.53 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0906 | 0.0334 | 0.0572 | 139.77 | 88.30 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2658 | 0.0985 | 0.1673 | 47.83 | 30.10 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0458 | 0.0189 | 0.0269 | 297.84 | 174.67 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0742 | 0.0181 | 0.0560 | 124.89 | 94.36 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1575 | 0.0559 | 0.1016 | 78.78 | 50.80 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2974 | 0.1287 | 0.1687 | 47.41 | 26.90 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2583 | 0.1110 | 0.1473 | 54.30 | 30.97 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2290 | 0.0848 | 0.1443 | 55.45 | 34.93 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2196 | 0.0731 | 0.1465 | 54.61 | 36.42 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6466 | 0.4442 | 0.2024 | 39.53 | 12.37 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0911 | 0.0300 | 0.0611 | 130.94 | 87.84 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1286 | 0.0383 | 0.0903 | 88.57 | 62.19 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1774 | 0.0910 | 0.0863 | 92.67 | 45.11 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1421 | 0.0549 | 0.0872 | 91.78 | 56.31 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1351 | 0.0598 | 0.0753 | 106.22 | 59.21 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3108 | 0.1275 | 0.1834 | 43.63 | 25.74 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2382 | 0.1922 | 0.0460 | 174.05 | 33.59 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2390 | 0.1914 | 0.0476 | 167.91 | 33.47 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3960 | 0.2906 | 0.1054 | 75.88 | 20.20 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4622 | 0.3144 | 0.1478 | 54.14 | 17.31 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2061 | 0.1144 | 0.0917 | 87.23 | 38.82 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2804 | 0.1467 | 0.1337 | 59.85 | 28.54 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1489 | 0.0605 | 0.0885 | 90.42 | 53.71 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0454 | 0.0171 | 0.0283 | 282.69 | 176.23 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4643 | 0.1725 | 0.2918 | 27.42 | 17.23 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0763 | 0.0157 | 0.0607 | 131.83 | 104.79 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1545 | 0.0792 | 0.0754 | 106.17 | 51.77 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0971 | 0.0431 | 0.0540 | 148.04 | 82.38 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0593 | 0.0147 | 0.0446 | 179.46 | 134.87 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1146 | 0.0388 | 0.0758 | 105.49 | 69.78 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6717 | 0.3848 | 0.2869 | 27.88 | 11.91 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2166 | 0.0767 | 0.1399 | 57.20 | 36.94 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3096 | 0.1225 | 0.1871 | 42.77 | 25.84 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2764 | 0.1043 | 0.1721 | 46.48 | 28.94 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3922 | 0.1690 | 0.2232 | 35.84 | 20.40 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1232 | 0.0455 | 0.0777 | 102.97 | 64.93 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0507 | 0.0104 | 0.0403 | 198.32 | 157.65 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2157 | 0.0718 | 0.1439 | 55.58 | 37.09 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0922 | 0.0411 | 0.0511 | 156.62 | 86.75 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2079 | 0.0919 | 0.1160 | 68.95 | 38.48 |
| `olmo3` | `olmo3` | 8 | 85 | 0.5018 | 0.2314 | 0.2703 | 29.60 | 15.94 |
| `apertus` | `apertus` | 8 | 76 | 0.5654 | 0.3203 | 0.2450 | 32.65 | 14.15 |

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
