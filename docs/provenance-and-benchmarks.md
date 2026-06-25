# Provenance and Benchmarks

Last updated: 2026-06-25

## Provenance

This repository is MIT licensed. Some files still carry source-port notes.
Keep those notes until each file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 0 | No Apple source notices remain in the audited paths. |
| Explicit source-port markers | 15 | Counted from real provenance markers, not ordinary comments that say "based on". |
| Files with no source-port marker | 79 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/MLXLLM/PhiMoE.swift` | Phi MoE attention layout, LongRoPE planning, router planning, guarded expert packing, stable checkpoint keys, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/SwitchLayers.swift` | Expert dispatch permutation, SwitchGLU routing, dense/quantized expert projection, and sorted-dispatch restoration. |
| `Sources/MLXLocalModels/MLXLLM/Granite.swift` | Granite attention layout, RoPE scaling plan, residual/embedding/logit scaling, tied/untied heads, greedy-token fast path, config defaults, stable checkpoint keys, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Ernie4_5.swift` | ERNIE 4.5 attention layout, explicit head-dimension override support, tied/untied heads, greedy-token fast path, config defaults, stable checkpoint keys, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Olmo3.swift` | OLMo3 sliding/full layer schedule, attention layout, q/k norm, YaRN-vs-sliding RoPE selection, cache layout, tied/untied heads, greedy-token fast path, sanitizer, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Qwen35.swift` | Qwen3.5 text config decoding, explicit layer schedule, attention and linear-attention layouts, cache planning, native MTP gating, tied/untied heads, greedy-token fast path, sanitizer, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Qwen35MoE.swift` | Qwen3.5 MoE top-level config fallback, shared top-level weight mapping, expert projection remapping, and sanitizer delegation. |
| `Sources/MLXLocalModels/MLXLLM/LFM2MoE.swift` | LFM2 MoE typed layer planning, attention/convolution layouts, router planning, guarded decoder dispatch, cache/KV-head planning, sanitizer packing, greedy path preservation, and LoRA target discovery. |
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
- Replaced OLMo3 with explicit sliding/full attention scheduling, q/k normalization, YaRN-vs-sliding RoPE selection, cache layout, tied/untied output handling, greedy-token fast path, and focused config/layout/cache/LoRA coverage.
- Replaced Qwen3.5 text and MoE wrappers with explicit config schedule decoding, validated layout plans, shared top-level weight mapping, native MTP gating, greedy-token fast path, and focused schedule/layout/cache/MTP coverage.
- Replaced LFM2 MoE with typed layer planning, explicit attention/convolution layouts, guarded layer dispatch, complete expert packing, attention-only LoRA targeting, and focused plan/cache/sanitizer coverage.
- Replaced Phi MoE with explicit attention and router plans, centralized LongRoPE support, safe expert packing, stable `model.*` parameter keys, greedy-token fast path, and focused config/layout/forward/sanitizer coverage.

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
`.build/benchmarks/test-all-architectures-2026-06-25-independent-phimoe.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0656 | 0.0308 | 0.0348 | 230.00 | 121.96 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1067 | 0.0437 | 0.0630 | 126.92 | 74.96 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1969 | 0.0824 | 0.1146 | 69.83 | 40.62 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3007 | 0.1200 | 0.1807 | 44.27 | 26.61 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0937 | 0.0369 | 0.0568 | 140.93 | 85.40 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2666 | 0.0991 | 0.1676 | 47.74 | 30.00 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0474 | 0.0188 | 0.0286 | 279.73 | 168.91 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0766 | 0.0337 | 0.0429 | 163.23 | 91.42 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1704 | 0.0904 | 0.0799 | 100.08 | 46.96 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2966 | 0.1361 | 0.1605 | 49.84 | 26.97 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.7887 | 0.5036 | 0.2851 | 28.06 | 10.14 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2572 | 0.1127 | 0.1445 | 55.35 | 31.10 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2265 | 0.0799 | 0.1466 | 54.56 | 35.31 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 1.2350 | 0.8169 | 0.4181 | 19.13 | 6.48 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0907 | 0.0254 | 0.0653 | 122.55 | 88.17 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1324 | 0.0420 | 0.0904 | 88.47 | 60.42 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1517 | 0.0654 | 0.0864 | 92.62 | 52.72 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1309 | 0.0479 | 0.0829 | 96.44 | 61.13 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1334 | 0.0623 | 0.0712 | 112.40 | 59.95 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2926 | 0.1187 | 0.1739 | 46.01 | 27.34 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2425 | 0.1954 | 0.0471 | 169.90 | 32.98 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2428 | 0.1968 | 0.0460 | 173.73 | 32.94 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3917 | 0.2863 | 0.1054 | 75.90 | 20.43 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4854 | 0.3375 | 0.1479 | 54.08 | 16.48 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2295 | 0.1407 | 0.0889 | 90.03 | 34.86 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.3006 | 0.1673 | 0.1333 | 60.00 | 26.61 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1474 | 0.0591 | 0.0883 | 90.59 | 54.27 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0476 | 0.0187 | 0.0289 | 276.84 | 168.19 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4956 | 0.2035 | 0.2922 | 27.38 | 16.14 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0744 | 0.0170 | 0.0574 | 139.45 | 107.57 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1833 | 0.0814 | 0.1019 | 78.50 | 43.64 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0934 | 0.0396 | 0.0538 | 148.63 | 85.63 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0613 | 0.0148 | 0.0465 | 172.19 | 130.50 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1155 | 0.0388 | 0.0767 | 104.27 | 69.27 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.8860 | 0.5791 | 0.3068 | 26.07 | 9.03 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2302 | 0.0909 | 0.1393 | 57.44 | 34.75 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2887 | 0.1017 | 0.1870 | 42.77 | 27.71 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.4728 | 0.3011 | 0.1717 | 46.60 | 16.92 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3922 | 0.1691 | 0.2231 | 35.86 | 20.40 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1228 | 0.0452 | 0.0776 | 103.07 | 65.16 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0510 | 0.0113 | 0.0397 | 201.29 | 156.74 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2237 | 0.0830 | 0.1407 | 56.85 | 35.76 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0982 | 0.0460 | 0.0522 | 153.21 | 81.46 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.1863 | 0.0817 | 0.1047 | 76.43 | 42.93 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4642 | 0.2106 | 0.2536 | 31.55 | 17.23 |
| `apertus` | `apertus` | 8 | 76 | 0.5564 | 0.3072 | 0.2492 | 32.10 | 14.38 |

## Skipped By Memory Gate

Presence is for the exact skipped model, not smaller siblings or different
quantization levels.

| Model | Reason | In `.build/test-models` | Other local copy found |
| --- | --- | --- | --- |
| `qwen3-moe-30b-a3b-4bit` | Requires 48 GiB RAM. | No. | No. |
| `mistral-small-24b-2501-4bit` | Requires 48 GiB RAM. | No. | No. |
| `phi-3.5-moe-instruct-4bit` | Requires 48 GiB RAM. | No. | No exact copy found in targeted checks on this host. |
| `gemma-3n-e2b-it-lm-bf16` | Requires 48 GiB RAM. | No. | No; only the 4-bit sibling is present. |
| `gemma-3n-e4b-it-lm-bf16` | Estimated runtime memory about 32 GiB; host budget is 24 GiB. | Yes, `gemma-3n-E4B-it-lm-bf16`. | No additional exact copy found. |
| `deepseek-r1-4bit` | Requires 256 GiB RAM. | No. | No. |
| `cohere-command-r-v01-4bit` | Estimated runtime memory about 53 GiB; host budget is 24 GiB. | Yes, `c4ai-command-r-v01-4bit`. | Yes, Patagonia/Think MLXSession resource copies. |
| `gpt-oss` | Requires 48 GiB RAM. | No. | No exact model directory found. |
| `qwen3-next` | Requires 64 GiB RAM. | No. | No. |
| `qwen3.5-moe` | Requires 48 GiB RAM. | No. | No; only the non-MoE `qwen3.5` test model is present. |
