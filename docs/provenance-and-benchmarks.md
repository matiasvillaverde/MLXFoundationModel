# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 5 | Keep notices until replaced. |
| Explicit port or based-on markers | 42 | Keep source notes until replaced. |
| Files with neither marker | 49 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-gemma.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0668 | 0.0313 | 0.0355 | 225.09 | 119.68 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1021 | 0.0402 | 0.0619 | 129.25 | 78.38 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1995 | 0.0847 | 0.1148 | 69.66 | 40.10 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3030 | 0.1209 | 0.1821 | 43.93 | 26.40 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0910 | 0.0345 | 0.0564 | 141.76 | 87.94 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2618 | 0.0945 | 0.1673 | 47.82 | 30.56 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0462 | 0.0180 | 0.0282 | 283.60 | 173.30 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0734 | 0.0190 | 0.0543 | 128.82 | 95.42 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1531 | 0.0515 | 0.1016 | 78.77 | 52.25 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2936 | 0.1255 | 0.1681 | 47.59 | 27.25 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2475 | 0.0995 | 0.1480 | 54.06 | 32.32 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2333 | 0.0881 | 0.1451 | 55.12 | 34.29 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2193 | 0.0736 | 0.1457 | 54.89 | 36.47 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6949 | 0.4911 | 0.2037 | 39.27 | 11.51 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0917 | 0.0298 | 0.0619 | 129.24 | 87.29 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1296 | 0.0405 | 0.0892 | 89.73 | 61.70 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1568 | 0.0702 | 0.0866 | 92.33 | 51.01 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1662 | 0.0828 | 0.0833 | 96.01 | 48.14 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1434 | 0.0673 | 0.0761 | 105.11 | 55.78 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3004 | 0.1189 | 0.1814 | 44.10 | 26.64 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2384 | 0.1920 | 0.0464 | 172.36 | 33.55 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2390 | 0.1925 | 0.0465 | 172.03 | 33.47 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3787 | 0.2732 | 0.1055 | 75.83 | 21.13 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4697 | 0.3219 | 0.1478 | 54.12 | 17.03 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2310 | 0.1403 | 0.0907 | 88.19 | 34.63 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2889 | 0.1552 | 0.1337 | 59.84 | 27.69 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1480 | 0.0598 | 0.0882 | 90.68 | 54.06 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0462 | 0.0173 | 0.0290 | 276.19 | 172.98 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4806 | 0.2010 | 0.2795 | 28.62 | 16.65 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0776 | 0.0161 | 0.0615 | 130.06 | 103.09 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1700 | 0.0791 | 0.0908 | 88.06 | 47.07 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0941 | 0.0402 | 0.0539 | 148.38 | 84.98 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0615 | 0.0154 | 0.0461 | 173.46 | 130.06 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1143 | 0.0363 | 0.0780 | 102.55 | 69.97 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6953 | 0.4076 | 0.2878 | 27.80 | 11.51 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2189 | 0.0794 | 0.1396 | 57.33 | 36.54 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2829 | 0.0965 | 0.1863 | 42.94 | 28.28 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2673 | 0.0945 | 0.1728 | 46.28 | 29.93 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3949 | 0.1726 | 0.2223 | 35.99 | 20.26 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1212 | 0.0433 | 0.0779 | 102.64 | 66.01 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0508 | 0.0105 | 0.0403 | 198.73 | 157.48 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2496 | 0.1082 | 0.1414 | 56.57 | 32.05 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0916 | 0.0427 | 0.0489 | 163.46 | 87.33 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2113 | 0.0975 | 0.1137 | 70.33 | 37.86 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4868 | 0.2190 | 0.2678 | 29.87 | 16.44 |
| `apertus` | `apertus` | 8 | 76 | 0.5793 | 0.3320 | 0.2473 | 32.35 | 13.81 |

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
