# Provenance and Benchmarks

Last updated: 2026-06-25

## Provenance

This repository is MIT licensed. Some files still carry source-port notes.
Keep those notes until each file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 0 | No Apple source notices remain in the audited paths. |
| Explicit source-port markers | 21 | Counted from real provenance markers, not ordinary comments that say "based on". |
| Files with no source-port marker | 73 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-25-independent-granite.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0655 | 0.0308 | 0.0347 | 230.83 | 122.14 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.0985 | 0.0352 | 0.0633 | 126.38 | 81.20 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1837 | 0.0669 | 0.1169 | 68.46 | 43.54 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3200 | 0.1379 | 0.1821 | 43.94 | 25.00 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0934 | 0.0344 | 0.0589 | 135.72 | 85.68 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2949 | 0.1273 | 0.1676 | 47.73 | 27.13 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0499 | 0.0197 | 0.0302 | 264.98 | 160.21 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0753 | 0.0317 | 0.0436 | 160.60 | 92.92 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1626 | 0.0828 | 0.0798 | 100.29 | 49.20 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3041 | 0.1438 | 0.1603 | 49.92 | 26.31 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2285 | 0.0812 | 0.1473 | 54.31 | 35.02 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2136 | 0.0695 | 0.1441 | 55.51 | 37.46 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2165 | 0.0701 | 0.1463 | 54.67 | 36.96 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.5880 | 0.3843 | 0.2037 | 39.27 | 13.61 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0977 | 0.0311 | 0.0666 | 120.14 | 81.89 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1307 | 0.0402 | 0.0905 | 88.43 | 61.21 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1473 | 0.0609 | 0.0864 | 92.65 | 54.31 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1370 | 0.0543 | 0.0827 | 96.70 | 58.38 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1398 | 0.0690 | 0.0708 | 113.00 | 57.22 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2963 | 0.1200 | 0.1763 | 45.37 | 27.00 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2398 | 0.1943 | 0.0455 | 175.99 | 33.37 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2433 | 0.1959 | 0.0474 | 168.61 | 32.88 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3812 | 0.2753 | 0.1059 | 75.55 | 20.99 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4602 | 0.3123 | 0.1479 | 54.11 | 17.38 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.1953 | 0.1107 | 0.0845 | 94.66 | 40.97 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2855 | 0.1546 | 0.1308 | 61.14 | 28.03 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1495 | 0.0615 | 0.0880 | 90.94 | 53.53 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0476 | 0.0178 | 0.0298 | 268.20 | 168.06 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4580 | 0.1660 | 0.2919 | 27.40 | 17.47 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0752 | 0.0159 | 0.0593 | 134.87 | 106.35 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1493 | 0.0847 | 0.0646 | 123.77 | 53.57 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0923 | 0.0377 | 0.0545 | 146.69 | 86.71 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0614 | 0.0146 | 0.0469 | 170.74 | 130.28 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1143 | 0.0384 | 0.0759 | 105.35 | 69.97 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6503 | 0.3709 | 0.2794 | 28.64 | 12.30 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2084 | 0.0690 | 0.1394 | 57.38 | 38.39 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3013 | 0.1141 | 0.1872 | 42.73 | 26.55 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2712 | 0.0986 | 0.1726 | 46.35 | 29.50 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3870 | 0.1634 | 0.2236 | 35.78 | 20.67 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1220 | 0.0449 | 0.0771 | 103.78 | 65.59 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0480 | 0.0103 | 0.0376 | 212.74 | 166.83 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2094 | 0.0681 | 0.1412 | 56.64 | 38.21 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0996 | 0.0445 | 0.0551 | 145.22 | 80.33 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2156 | 0.1001 | 0.1155 | 69.26 | 37.11 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4913 | 0.2215 | 0.2697 | 29.66 | 16.28 |
| `apertus` | `apertus` | 8 | 76 | 0.5798 | 0.3345 | 0.2454 | 32.61 | 13.80 |

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
