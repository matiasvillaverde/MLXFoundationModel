# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 2 | Keep notices until replaced. |
| Explicit port or based-on markers | 40 | Keep source notes until replaced. |
| Files with neither marker | 52 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/MLXLLM/Llama.swift` | Llama/Mistral attention layout, linear/dynamic/Llama 3 RoPE planning, decoder block, backbone, tied/untied output heads, greedy-token fast path, config validation, and LoRA target discovery. |
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
- Replaced Llama with explicit Llama/Mistral layout, project-owned RoPE planning for linear, dynamic, and Llama 3 scaling, tied/untied output handling, greedy-token fast path, and focused config/layout/LoRA coverage.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-llama.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0685 | 0.0325 | 0.0360 | 221.92 | 116.76 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.0999 | 0.0366 | 0.0632 | 126.49 | 80.12 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.2067 | 0.0920 | 0.1147 | 69.73 | 38.70 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3035 | 0.1214 | 0.1821 | 43.94 | 26.36 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0943 | 0.0350 | 0.0593 | 134.85 | 84.82 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2739 | 0.1067 | 0.1671 | 47.86 | 29.21 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0468 | 0.0192 | 0.0276 | 290.37 | 171.09 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0773 | 0.0332 | 0.0441 | 158.75 | 90.60 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1572 | 0.0778 | 0.0795 | 100.65 | 50.88 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2885 | 0.1283 | 0.1603 | 49.92 | 27.72 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2480 | 0.1002 | 0.1479 | 54.10 | 32.25 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2311 | 0.0870 | 0.1441 | 55.52 | 34.62 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2260 | 0.0800 | 0.1460 | 54.80 | 35.40 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6963 | 0.4928 | 0.2035 | 39.31 | 11.49 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0894 | 0.0269 | 0.0626 | 127.87 | 89.45 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1326 | 0.0420 | 0.0906 | 88.33 | 60.33 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1438 | 0.0566 | 0.0871 | 91.81 | 55.65 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1575 | 0.0718 | 0.0857 | 93.34 | 50.78 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1521 | 0.0811 | 0.0710 | 112.74 | 52.61 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2999 | 0.1230 | 0.1769 | 45.21 | 26.68 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2448 | 0.1976 | 0.0472 | 169.52 | 32.68 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2443 | 0.1978 | 0.0465 | 172.03 | 32.74 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3802 | 0.2750 | 0.1053 | 76.00 | 21.04 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4738 | 0.3203 | 0.1534 | 52.14 | 16.89 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2015 | 0.1182 | 0.0833 | 95.98 | 39.70 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2898 | 0.1562 | 0.1336 | 59.89 | 27.61 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1454 | 0.0568 | 0.0886 | 90.30 | 55.02 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0463 | 0.0173 | 0.0289 | 276.67 | 172.95 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4421 | 0.1652 | 0.2769 | 28.89 | 18.10 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0772 | 0.0157 | 0.0615 | 130.06 | 103.61 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1731 | 0.0913 | 0.0817 | 97.88 | 46.22 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0947 | 0.0400 | 0.0547 | 146.27 | 84.49 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0619 | 0.0149 | 0.0470 | 170.09 | 129.18 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1158 | 0.0398 | 0.0760 | 105.24 | 69.07 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.7280 | 0.4357 | 0.2923 | 27.37 | 10.99 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2111 | 0.0714 | 0.1397 | 57.26 | 37.90 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3375 | 0.1503 | 0.1872 | 42.73 | 23.71 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2721 | 0.1004 | 0.1717 | 46.59 | 29.40 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.4046 | 0.1813 | 0.2233 | 35.83 | 19.77 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1224 | 0.0448 | 0.0776 | 103.10 | 65.35 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0472 | 0.0104 | 0.0368 | 217.32 | 169.37 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2262 | 0.0849 | 0.1413 | 56.61 | 35.36 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0938 | 0.0446 | 0.0492 | 162.67 | 85.27 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2056 | 0.0916 | 0.1140 | 70.16 | 38.91 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4944 | 0.2239 | 0.2705 | 29.58 | 16.18 |
| `apertus` | `apertus` | 8 | 76 | 0.5469 | 0.3009 | 0.2460 | 32.52 | 14.63 |

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
