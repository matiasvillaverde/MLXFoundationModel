# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 10 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 45 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-lora-train.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0687 | 0.0327 | 0.0360 | 222.32 | 116.51 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1012 | 0.0380 | 0.0632 | 126.66 | 79.08 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1812 | 0.0645 | 0.1167 | 68.58 | 44.15 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3015 | 0.1192 | 0.1823 | 43.89 | 26.54 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0936 | 0.0367 | 0.0569 | 140.68 | 85.46 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2619 | 0.0944 | 0.1675 | 47.76 | 30.55 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0479 | 0.0193 | 0.0286 | 279.33 | 166.96 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0711 | 0.0182 | 0.0530 | 132.16 | 98.41 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1579 | 0.0568 | 0.1011 | 79.13 | 50.67 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3325 | 0.1637 | 0.1688 | 47.39 | 24.06 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2529 | 0.1056 | 0.1474 | 54.28 | 31.63 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2297 | 0.0864 | 0.1434 | 55.80 | 34.82 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2164 | 0.0701 | 0.1463 | 54.68 | 36.96 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.6525 | 0.4487 | 0.2038 | 39.26 | 12.26 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0911 | 0.0300 | 0.0611 | 130.93 | 87.85 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1311 | 0.0405 | 0.0905 | 88.38 | 61.04 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1945 | 0.1084 | 0.0860 | 92.99 | 41.14 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1354 | 0.0478 | 0.0875 | 91.38 | 59.09 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1373 | 0.0618 | 0.0755 | 105.94 | 58.27 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3136 | 0.1341 | 0.1795 | 44.58 | 25.51 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2385 | 0.1912 | 0.0473 | 169.11 | 33.55 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2396 | 0.1934 | 0.0462 | 173.12 | 33.39 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3906 | 0.2740 | 0.1166 | 68.60 | 20.48 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4697 | 0.3235 | 0.1463 | 54.69 | 17.03 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2435 | 0.1397 | 0.1038 | 77.11 | 32.85 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2912 | 0.1578 | 0.1333 | 60.00 | 27.48 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1463 | 0.0578 | 0.0884 | 90.47 | 54.70 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0489 | 0.0184 | 0.0305 | 262.23 | 163.54 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4460 | 0.1663 | 0.2797 | 28.60 | 17.94 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0765 | 0.0159 | 0.0606 | 132.00 | 104.62 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1686 | 0.0808 | 0.0878 | 91.14 | 47.44 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0926 | 0.0390 | 0.0536 | 149.26 | 86.39 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0617 | 0.0147 | 0.0470 | 170.25 | 129.64 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1111 | 0.0345 | 0.0766 | 104.41 | 71.99 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6586 | 0.3797 | 0.2789 | 28.69 | 12.15 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2093 | 0.0695 | 0.1399 | 57.19 | 38.22 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2954 | 0.1081 | 0.1873 | 42.71 | 27.08 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2784 | 0.1056 | 0.1728 | 46.29 | 28.73 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3578 | 0.1364 | 0.2214 | 36.14 | 22.36 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1222 | 0.0446 | 0.0777 | 103.02 | 65.44 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0482 | 0.0105 | 0.0377 | 212.19 | 165.99 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2265 | 0.0825 | 0.1440 | 55.54 | 35.32 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0960 | 0.0464 | 0.0496 | 161.41 | 83.36 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2133 | 0.0977 | 0.1157 | 69.17 | 37.50 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4783 | 0.2101 | 0.2682 | 29.83 | 16.73 |
| `apertus` | `apertus` | 8 | 76 | 0.5546 | 0.3101 | 0.2444 | 32.73 | 14.43 |

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
