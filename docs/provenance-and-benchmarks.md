# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files still note ports from MLX model
implementations. Keep those source notes until each file is replaced by an
independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 0 | No Apple source notices remain in the audited paths. |
| Explicit port or based-on markers | 29 | Keep source notes until replaced. |
| Files with neither marker | 65 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/Common/KVCache.swift` | Cache protocols, dense/rotating/chunked/quantized KV storage, prompt-cache serialization, quantized attention, and runtime cache quantization. |
| `Sources/MLXLocalModels/Common/LanguageModel.swift` | Core model input/output contracts, default forwarding, greedy helpers, and cache creation. |
| `Sources/MLXLocalModels/Common/Load.swift` | Model artifact matching, deterministic safetensor discovery, and weight loading. |
| `Sources/MLXLocalModels/Common/LoRA+Layers.swift` | Dense and quantized LoRA replacement layers, adapter initialization, freeze policy, and fusion. |
| `Sources/MLXLocalModels/Common/Module+Extensions.swift` | Logical parameter counting for dense and quantized modules. |
| `Sources/MLXLocalModels/Common/RoPEApplication.swift` | Scalar and batch RoPE offset selection for KV caches. |
| `Sources/MLXLocalModels/Common/SuScaledRoPE.swift` | LongRoPE factor planning, short/long frequency selection, scalar and batch offsets, and non-rotary tail preservation. |
| `Sources/MLXLocalModels/Common/Tokenizer.swift` | Tokenizer loading, tokenizer-class rewriting, replacement registry, and streaming detokenization. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |
| `Sources/MLXLocalModels/MLXLLM/LLMModelFactory.swift` | LLM type registration, alias grouping, model load progress, generation-token resolution, and trampoline factory. |
| `Sources/MLXLocalModels/MLXLLM/Gemma.swift` | Gemma RMSNorm, residual clipping, attention layout, decoder block, backbone, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Gemma2.swift` | Gemma2 soft-capped attention, grouped KV expansion, decoder block, backbone, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Internlm2.swift` | InternLM2 packed attention, dynamic RoPE planning, decoder blocks, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/DeepseekV3.swift` | DeepSeek V3 attention layout, YaRN planning, grouped MoE routing, checkpoint key normalization, cache dimensions, greedy-token fast path, sanitizer packing, and LoRA target discovery. |
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
- Replaced Su-scaled RoPE with explicit LongRoPE planning, short/long factor validation, context-length frequency selection, and focused plan/call coverage.
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
- Replaced KV cache internals with shared append planning, explicit dense and quantized state wrappers, active-window chunk trimming, stable prompt-cache layout serialization, and focused cache growth/chunk/quantization coverage.
- Replaced Phi with an explicit attention layout, project-owned module structure, config defaults, greedy-token fast path, and focused layout/config/LoRA coverage.
- Replaced InternLM2 with packed-attention layout, type-specific RoPE scaling, greedy-token fast path, packed LoRA targeting, and focused layout/RoPE/config coverage.
- Replaced Gemma with a shared project-owned norm, explicit attention layout, stable checkpoint keys, greedy-token fast path, and focused config/layout/LoRA coverage.
- Replaced Gemma2 with soft-capped attention layout, grouped KV expansion, greedy-token fast path, stable checkpoint keys, and focused config/layout/LoRA coverage.
- Replaced Phi3 with packed QKV layout, explicit RoPE/LongRoPE planning, tied/untied output handling, greedy-token fast path, and focused config/layout/LoRA coverage.
- Replaced Llama with explicit Llama/Mistral layout, project-owned RoPE planning for linear, dynamic, and Llama 3 scaling, tied/untied output handling, greedy-token fast path, and focused config/layout/LoRA coverage.
- Replaced DeepSeek V3 with explicit attention, YaRN, and MoE routing plans; fixed empty KV-cache dimensions; corrected adapter targets; packed expert weights in the sanitizer; and added focused config/layout/routing/forward/sanitizer coverage.

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

The selected `deepseek-r1-distill-qwen-7b-4bit` checkpoint is cataloged as
`deepseek_v3`, but its local `config.json` declares `model_type: qwen2`. It is
kept in the table as a catalog regression check, not as proof of the true
DeepSeek V3 MoE implementation. The full `deepseek-r1-4bit` checkpoint was
skipped by the memory gate and was not found locally.

## Benchmarks

These rows come from `BENCH` lines printed by the real-model test runner in
`.build/benchmarks/test-all-architectures-2026-06-24-independent-suscaledrope.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0672 | 0.0316 | 0.0356 | 224.94 | 119.12 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1027 | 0.0409 | 0.0618 | 129.38 | 77.91 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1843 | 0.0675 | 0.1168 | 68.48 | 43.40 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2930 | 0.1114 | 0.1815 | 44.07 | 27.31 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0944 | 0.0377 | 0.0567 | 141.06 | 84.72 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2749 | 0.1070 | 0.1680 | 47.63 | 29.10 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0445 | 0.0182 | 0.0263 | 304.64 | 179.93 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0755 | 0.0325 | 0.0430 | 162.81 | 92.73 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1575 | 0.0777 | 0.0798 | 100.22 | 50.80 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3050 | 0.1448 | 0.1602 | 49.94 | 26.23 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2401 | 0.0922 | 0.1479 | 54.08 | 33.32 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2138 | 0.0707 | 0.1431 | 55.91 | 37.42 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2177 | 0.0711 | 0.1467 | 54.54 | 36.74 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6237 | 0.4208 | 0.2029 | 39.43 | 12.83 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0936 | 0.0291 | 0.0645 | 124.05 | 85.46 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1316 | 0.0411 | 0.0905 | 88.38 | 60.77 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1351 | 0.0495 | 0.0856 | 93.41 | 59.20 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1467 | 0.0620 | 0.0847 | 94.47 | 54.53 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1252 | 0.0535 | 0.0717 | 111.62 | 63.91 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2954 | 0.1199 | 0.1755 | 45.59 | 27.08 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2412 | 0.1957 | 0.0455 | 175.94 | 33.17 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2375 | 0.1921 | 0.0454 | 176.02 | 33.68 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3773 | 0.2702 | 0.1072 | 74.65 | 21.20 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4823 | 0.3347 | 0.1476 | 54.20 | 16.59 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2067 | 0.1232 | 0.0835 | 95.77 | 38.69 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2820 | 0.1484 | 0.1336 | 59.88 | 28.37 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1498 | 0.0613 | 0.0886 | 90.34 | 53.39 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0462 | 0.0172 | 0.0290 | 276.28 | 173.29 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4860 | 0.2066 | 0.2794 | 28.63 | 16.46 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0750 | 0.0159 | 0.0591 | 135.39 | 106.68 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1470 | 0.0824 | 0.0646 | 123.75 | 54.41 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0953 | 0.0412 | 0.0541 | 147.78 | 83.95 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0615 | 0.0147 | 0.0468 | 170.88 | 130.06 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1125 | 0.0356 | 0.0768 | 104.11 | 71.14 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6251 | 0.3446 | 0.2805 | 28.52 | 12.80 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2073 | 0.0672 | 0.1401 | 57.12 | 38.60 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2843 | 0.0969 | 0.1874 | 42.69 | 28.14 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2785 | 0.1061 | 0.1724 | 46.41 | 28.73 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3677 | 0.1446 | 0.2231 | 35.85 | 21.75 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1280 | 0.0506 | 0.0774 | 103.34 | 62.51 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0506 | 0.0103 | 0.0403 | 198.69 | 158.08 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2236 | 0.0822 | 0.1414 | 56.58 | 35.78 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0963 | 0.0451 | 0.0512 | 156.37 | 83.08 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2195 | 0.1056 | 0.1138 | 70.28 | 36.45 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4837 | 0.2128 | 0.2709 | 29.53 | 16.54 |
| `apertus` | `apertus` | 8 | 76 | 0.5637 | 0.3203 | 0.2434 | 32.87 | 14.19 |

## Skipped By Memory Gate

Presence is for the exact skipped model, not smaller siblings or different
quantization levels.

| Model | Reason | In `.build/test-models` | Other local copy found |
| --- | --- | --- | --- |
| `qwen3-moe-30b-a3b-4bit` | Requires 48 GiB RAM. | No. | No. |
| `mistral-small-24b-2501-4bit` | Requires 48 GiB RAM. | No. | No. |
| `phi-3.5-moe-instruct-4bit` | Requires 48 GiB RAM. | No. | Yes, `Patagonia-client/MLXSession/Tests/MLXPhiMoETests/Resources/Phi-3.5-MoE-instruct-4bit`. |
| `gemma-3n-e2b-it-lm-bf16` | Requires 48 GiB RAM. | No. | No; only the 4-bit sibling is present. |
| `gemma-3n-e4b-it-lm-bf16` | Estimated runtime memory about 32 GiB; host budget is 24 GiB. | Yes, `gemma-3n-E4B-it-lm-bf16`. | No additional exact copy found. |
| `deepseek-r1-4bit` | Requires 256 GiB RAM. | No. | No. |
| `cohere-command-r-v01-4bit` | Estimated runtime memory about 53 GiB; host budget is 24 GiB. | Yes, `c4ai-command-r-v01-4bit`. | Yes, Patagonia/Think MLXSession resource copies. |
| `gpt-oss` | Requires 48 GiB RAM. | No. | No exact model directory found. |
| `qwen3-next` | Requires 64 GiB RAM. | No. | No. |
| `qwen3.5-moe` | Requires 48 GiB RAM. | No. | No; only the non-MoE `qwen3.5` test model is present. |
