# Provenance and Benchmarks

Last updated: 2026-06-25

## Provenance

This repository is MIT licensed. Some files still carry source-port notes.
Keep those notes until each file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 0 | No Apple source notices remain in the audited paths. |
| Explicit source-port markers | 20 | Counted from real provenance markers, not ordinary comments that say "based on". |
| Files with no source-port marker | 74 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/MLXLLM/SwitchLayers.swift` | Expert dispatch permutation, SwitchGLU routing, dense/quantized expert projection, and sorted-dispatch restoration. |
| `Sources/MLXLocalModels/MLXLLM/Granite.swift` | Granite attention layout, RoPE scaling plan, residual/embedding/logit scaling, tied/untied heads, greedy-token fast path, config defaults, stable checkpoint keys, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Ernie4_5.swift` | ERNIE 4.5 attention layout, explicit head-dimension override support, tied/untied heads, greedy-token fast path, config defaults, stable checkpoint keys, and LoRA target discovery. |
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
- Replaced SwitchLayers with explicit expert routing permutations, shared dense/quantized bias handling, deterministic sorted-dispatch coverage, and real MoE validation.
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
- Replaced Granite with an explicit attention layout, linear RoPE scaling plan, stable checkpoint-compatible `model.*` parameter keys, tied/untied output handling, greedy-token fast path, and focused config/layout/forward/LoRA coverage.
- Replaced ERNIE 4.5 with an explicit attention layout, head-dimension fallback and override handling, stable checkpoint-compatible `model.*` parameter keys, tied/untied output handling, greedy-token fast path, and focused config/layout/forward/LoRA coverage.

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
`.build/benchmarks/test-all-architectures-2026-06-25-independent-ernie45.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0666 | 0.0312 | 0.0354 | 226.03 | 120.11 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1007 | 0.0375 | 0.0633 | 126.42 | 79.41 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1933 | 0.0764 | 0.1168 | 68.47 | 41.40 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2873 | 0.1051 | 0.1822 | 43.90 | 27.84 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0909 | 0.0342 | 0.0567 | 141.21 | 88.02 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2879 | 0.1206 | 0.1674 | 47.80 | 27.78 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0457 | 0.0192 | 0.0265 | 301.95 | 175.10 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0755 | 0.0321 | 0.0434 | 161.20 | 92.71 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1816 | 0.1023 | 0.0793 | 100.89 | 44.06 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2992 | 0.1392 | 0.1600 | 49.99 | 26.73 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2514 | 0.1034 | 0.1480 | 54.05 | 31.82 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2170 | 0.0728 | 0.1442 | 55.48 | 36.86 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2198 | 0.0733 | 0.1465 | 54.60 | 36.39 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6705 | 0.4669 | 0.2036 | 39.30 | 11.93 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0914 | 0.0294 | 0.0620 | 129.00 | 87.54 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1278 | 0.0374 | 0.0904 | 88.49 | 62.61 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1493 | 0.0636 | 0.0856 | 93.41 | 53.60 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1331 | 0.0484 | 0.0847 | 94.43 | 60.11 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1509 | 0.0688 | 0.0821 | 97.46 | 53.01 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2982 | 0.1211 | 0.1771 | 45.17 | 26.83 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2418 | 0.1962 | 0.0456 | 175.51 | 33.08 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2402 | 0.1937 | 0.0465 | 172.05 | 33.31 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3885 | 0.2831 | 0.1054 | 75.88 | 20.59 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4842 | 0.3297 | 0.1545 | 51.79 | 16.52 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2052 | 0.1140 | 0.0912 | 87.73 | 38.99 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2923 | 0.1610 | 0.1313 | 60.94 | 27.37 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1483 | 0.0598 | 0.0884 | 90.47 | 53.96 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0464 | 0.0167 | 0.0296 | 270.18 | 172.59 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4864 | 0.1943 | 0.2922 | 27.38 | 16.45 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0760 | 0.0160 | 0.0601 | 133.17 | 105.20 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1443 | 0.0722 | 0.0721 | 110.88 | 55.44 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0949 | 0.0386 | 0.0562 | 142.29 | 84.33 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0601 | 0.0149 | 0.0452 | 176.97 | 133.09 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1124 | 0.0357 | 0.0768 | 104.20 | 71.14 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.5710 | 0.2974 | 0.2736 | 29.24 | 14.01 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2136 | 0.0737 | 0.1399 | 57.20 | 37.46 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2899 | 0.1026 | 0.1873 | 42.72 | 27.60 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2755 | 0.1027 | 0.1728 | 46.30 | 29.04 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3579 | 0.1357 | 0.2223 | 35.99 | 22.35 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1236 | 0.0468 | 0.0767 | 104.28 | 64.75 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0509 | 0.0106 | 0.0403 | 198.41 | 157.18 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2110 | 0.0698 | 0.1412 | 56.64 | 37.91 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.1003 | 0.0503 | 0.0499 | 160.20 | 79.79 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2117 | 0.0980 | 0.1137 | 70.35 | 37.79 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4870 | 0.2172 | 0.2698 | 29.65 | 16.43 |
| `apertus` | `apertus` | 8 | 76 | 0.5565 | 0.3123 | 0.2441 | 32.77 | 14.38 |

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
