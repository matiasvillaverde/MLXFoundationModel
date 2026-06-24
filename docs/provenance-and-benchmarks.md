# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 1 | Keep notices until replaced. |
| Explicit port or based-on markers | 40 | Keep source notes until replaced. |
| Files with neither marker | 53 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-deepseekv3.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0674 | 0.0319 | 0.0356 | 224.86 | 118.61 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.0993 | 0.0361 | 0.0632 | 126.56 | 80.59 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1852 | 0.0683 | 0.1169 | 68.46 | 43.20 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2942 | 0.1140 | 0.1802 | 44.38 | 27.19 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0933 | 0.0365 | 0.0568 | 140.83 | 85.78 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2556 | 0.0882 | 0.1674 | 47.80 | 31.30 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0474 | 0.0196 | 0.0278 | 287.75 | 168.92 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0762 | 0.0326 | 0.0435 | 160.74 | 91.90 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1688 | 0.0886 | 0.0802 | 99.76 | 47.41 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2929 | 0.1327 | 0.1603 | 49.92 | 27.31 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2421 | 0.0940 | 0.1481 | 54.02 | 33.05 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2142 | 0.0711 | 0.1430 | 55.93 | 37.36 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2191 | 0.0725 | 0.1466 | 54.58 | 36.52 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.5331 | 0.3270 | 0.2061 | 38.82 | 15.01 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0899 | 0.0266 | 0.0633 | 126.41 | 88.98 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1344 | 0.0437 | 0.0907 | 88.25 | 59.53 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1551 | 0.0684 | 0.0867 | 92.29 | 51.58 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1803 | 0.0991 | 0.0812 | 98.49 | 44.38 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1425 | 0.0717 | 0.0708 | 112.96 | 56.13 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2828 | 0.1058 | 0.1770 | 45.19 | 28.29 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2396 | 0.1942 | 0.0454 | 176.20 | 33.39 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2490 | 0.2027 | 0.0463 | 172.73 | 32.13 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3871 | 0.2818 | 0.1053 | 75.98 | 20.67 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4650 | 0.3169 | 0.1481 | 54.03 | 17.21 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.1891 | 0.1056 | 0.0836 | 95.75 | 42.30 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.3317 | 0.1986 | 0.1330 | 60.13 | 24.12 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1498 | 0.0612 | 0.0887 | 90.24 | 53.39 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0472 | 0.0179 | 0.0293 | 273.10 | 169.59 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4600 | 0.1695 | 0.2905 | 27.54 | 17.39 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0769 | 0.0175 | 0.0593 | 134.80 | 104.08 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1394 | 0.0722 | 0.0673 | 118.95 | 57.38 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0932 | 0.0392 | 0.0540 | 148.25 | 85.84 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0610 | 0.0154 | 0.0456 | 175.32 | 131.16 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1147 | 0.0382 | 0.0766 | 104.48 | 69.73 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.4896 | 0.2144 | 0.2752 | 29.07 | 16.34 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2094 | 0.0694 | 0.1400 | 57.16 | 38.21 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3223 | 0.1374 | 0.1849 | 43.27 | 24.82 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2948 | 0.1220 | 0.1728 | 46.31 | 27.14 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3674 | 0.1437 | 0.2237 | 35.77 | 21.78 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1239 | 0.0471 | 0.0768 | 104.21 | 64.56 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0513 | 0.0114 | 0.0399 | 200.32 | 155.88 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2157 | 0.0760 | 0.1397 | 57.29 | 37.09 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0982 | 0.0475 | 0.0507 | 157.78 | 81.44 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2028 | 0.0887 | 0.1140 | 70.15 | 39.45 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4889 | 0.2204 | 0.2686 | 29.79 | 16.36 |
| `apertus` | `apertus` | 8 | 76 | 0.5541 | 0.3086 | 0.2455 | 32.59 | 14.44 |

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
