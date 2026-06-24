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
| Explicit port or based-on markers | 30 | Keep source notes until replaced. |
| Files with neither marker | 64 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-kvcache.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0688 | 0.0324 | 0.0364 | 219.93 | 116.27 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1015 | 0.0385 | 0.0630 | 126.97 | 78.82 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1829 | 0.0660 | 0.1169 | 68.42 | 43.75 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2914 | 0.1096 | 0.1818 | 44.00 | 27.46 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0929 | 0.0367 | 0.0562 | 142.42 | 86.10 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2875 | 0.1198 | 0.1677 | 47.71 | 27.83 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0449 | 0.0176 | 0.0272 | 293.69 | 178.26 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0769 | 0.0331 | 0.0437 | 160.05 | 91.05 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1582 | 0.0788 | 0.0795 | 100.68 | 50.56 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2928 | 0.1339 | 0.1589 | 50.35 | 27.32 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2536 | 0.1056 | 0.1480 | 54.07 | 31.55 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2165 | 0.0720 | 0.1445 | 55.38 | 36.95 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2197 | 0.0740 | 0.1457 | 54.92 | 36.41 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6084 | 0.4058 | 0.2026 | 39.49 | 13.15 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0933 | 0.0301 | 0.0632 | 126.49 | 85.72 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1344 | 0.0446 | 0.0898 | 89.12 | 59.52 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1438 | 0.0577 | 0.0861 | 92.89 | 55.62 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1284 | 0.0427 | 0.0856 | 93.42 | 62.32 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1261 | 0.0548 | 0.0713 | 112.14 | 63.44 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2886 | 0.1114 | 0.1773 | 45.12 | 27.72 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2421 | 0.1968 | 0.0452 | 176.80 | 33.05 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2434 | 0.1962 | 0.0471 | 169.76 | 32.87 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3851 | 0.2796 | 0.1054 | 75.87 | 20.77 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4684 | 0.3130 | 0.1554 | 51.49 | 17.08 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2018 | 0.1176 | 0.0842 | 94.98 | 39.64 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2999 | 0.1668 | 0.1331 | 60.09 | 26.67 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1449 | 0.0565 | 0.0884 | 90.50 | 55.20 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0458 | 0.0172 | 0.0286 | 279.72 | 174.67 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4803 | 0.2012 | 0.2792 | 28.66 | 16.66 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0739 | 0.0155 | 0.0583 | 137.12 | 108.30 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1375 | 0.0735 | 0.0640 | 125.05 | 58.18 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0945 | 0.0398 | 0.0547 | 146.27 | 84.64 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0606 | 0.0149 | 0.0458 | 174.75 | 131.92 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1111 | 0.0352 | 0.0759 | 105.35 | 72.00 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.5007 | 0.2221 | 0.2786 | 28.71 | 15.98 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2082 | 0.0683 | 0.1400 | 57.15 | 38.42 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2938 | 0.1065 | 0.1872 | 42.73 | 27.23 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2762 | 0.1048 | 0.1713 | 46.69 | 28.97 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3614 | 0.1380 | 0.2234 | 35.82 | 22.14 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1216 | 0.0439 | 0.0777 | 102.99 | 65.81 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0506 | 0.0105 | 0.0401 | 199.44 | 158.19 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2145 | 0.0741 | 0.1405 | 56.96 | 37.30 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0958 | 0.0462 | 0.0496 | 161.15 | 83.47 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2340 | 0.1206 | 0.1133 | 70.58 | 34.19 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4958 | 0.2247 | 0.2711 | 29.51 | 16.14 |
| `apertus` | `apertus` | 8 | 76 | 0.5566 | 0.3123 | 0.2443 | 32.75 | 14.37 |

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
