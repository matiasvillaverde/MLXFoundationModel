# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 14 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 41 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |
| `Sources/MLXLocalModels/Common/ModelConfiguration.swift` | Model identity, tokenizer overrides, local directory resolution, and generation token defaults. |
| `Sources/MLXLocalModels/Common/ModelContainer.swift` | Actor-owned model context and prompt-cache access. |
| `Sources/MLXLocalModels/Common/AbstractModelRegistry.swift` | Thread-safe model configuration lookup and fallback creation. |
| `Sources/MLXLocalModels/Common/ModelTypeRegistry.swift` | Thread-safe model type constructor lookup and unsupported-type reporting. |
| `Sources/MLXLocalModels/Common/GenerationConstants.swift` | Shared generation, cache, and sampling defaults. |
| `Sources/MLXLocalModels/Common/LanguageModel.swift` | Core model input/output contracts, default forwarding, greedy helpers, and cache creation. |
| `Sources/MLXLocalModels/Common/Module+Extensions.swift` | Logical parameter counting for dense and quantized modules. |
| `Sources/MLXLocalModels/Common/RoPEApplication.swift` | Scalar and batch RoPE offset selection for KV caches. |
| `Sources/MLXLocalModels/Common/Tokenizer.swift` | Tokenizer loading, tokenizer-class rewriting, replacement registry, and streaming detokenization. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |
| `Sources/MLXLocalModels/MLXLLM/Lora+Data.swift` | LoRA JSONL/text data lookup and parsing. |

## Recent Code Changes

Current independence pass:

- Replaced the shared model configuration and registry files listed above.
- Removed an unused prompt-preparation closure from `ModelConfiguration`; its only registry call site never affected runtime behavior because the closure was not stored or used.
- Added focused SwiftTesting coverage for remote and local model identity, equality, fallback configuration creation, replacement registration, constructor lookup, unsupported model errors, and concurrent registry writes.
- Replaced default text-model prefill support with focused coverage for constants, default chunk size, explicit window chunking, and prompt-tail preservation.
- Replaced module parameter counting with a single-pass implementation and focused coverage for dense, embedding, and quantized module leaves.
- Replaced RoPE offset selection with focused coverage for nil-cache, scalar-cache, and per-row batch-cache paths.
- Replaced model container ownership with focused coverage for context updates, perform forwarding, legacy overload compatibility, and prompt-cache mutation.
- Replaced tokenizer support with focused coverage for tokenizer-class rewriting, registry updates, streaming deltas, newline resets, and incomplete Unicode boundaries.
- Replaced LoRA data loading with focused coverage for lookup precedence, JSONL parsing, text lines, missing files, and unsupported file types.
- Replaced LanguageModel core contracts with focused coverage for input slicing, media wrappers, default forwarding, greedy helpers, sanitize fallback, and KV-cache creation.

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-language-model.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0644 | 0.0297 | 0.0347 | 230.30 | 124.24 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1002 | 0.0365 | 0.0637 | 125.54 | 79.86 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1810 | 0.0642 | 0.1168 | 68.48 | 44.20 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3032 | 0.1209 | 0.1823 | 43.88 | 26.39 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0929 | 0.0361 | 0.0568 | 140.85 | 86.12 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2729 | 0.1052 | 0.1677 | 47.72 | 29.32 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0451 | 0.0178 | 0.0273 | 293.14 | 177.41 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0743 | 0.0183 | 0.0561 | 124.88 | 94.17 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1554 | 0.0538 | 0.1016 | 78.73 | 51.48 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2884 | 0.1196 | 0.1689 | 47.38 | 27.74 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2526 | 0.1047 | 0.1479 | 54.07 | 31.67 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2297 | 0.0852 | 0.1445 | 55.36 | 34.82 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2235 | 0.0779 | 0.1455 | 54.97 | 35.80 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.7044 | 0.5007 | 0.2037 | 39.26 | 11.36 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0907 | 0.0282 | 0.0625 | 127.98 | 88.16 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1306 | 0.0399 | 0.0907 | 88.17 | 61.25 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1825 | 0.0960 | 0.0865 | 92.49 | 43.84 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1257 | 0.0379 | 0.0878 | 91.10 | 63.64 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1543 | 0.0800 | 0.0743 | 107.66 | 51.85 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.2985 | 0.1148 | 0.1837 | 43.55 | 26.80 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2404 | 0.1946 | 0.0458 | 174.72 | 33.27 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2470 | 0.1986 | 0.0484 | 165.31 | 32.38 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3843 | 0.2759 | 0.1084 | 73.79 | 20.81 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4724 | 0.3247 | 0.1477 | 54.15 | 16.93 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2023 | 0.1142 | 0.0881 | 90.83 | 39.55 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2949 | 0.1610 | 0.1339 | 59.75 | 27.13 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1465 | 0.0589 | 0.0877 | 91.25 | 54.60 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0454 | 0.0178 | 0.0276 | 290.03 | 176.25 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4343 | 0.1661 | 0.2682 | 29.83 | 18.42 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0776 | 0.0159 | 0.0617 | 129.74 | 103.09 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1855 | 0.0912 | 0.0943 | 84.85 | 43.14 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0937 | 0.0390 | 0.0547 | 146.33 | 85.40 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0608 | 0.0146 | 0.0462 | 173.21 | 131.62 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1110 | 0.0345 | 0.0765 | 104.56 | 72.05 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6618 | 0.3860 | 0.2758 | 29.01 | 12.09 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2157 | 0.0760 | 0.1397 | 57.25 | 37.09 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3053 | 0.1182 | 0.1871 | 42.75 | 26.20 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2751 | 0.1025 | 0.1727 | 46.33 | 29.08 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3705 | 0.1478 | 0.2227 | 35.92 | 21.59 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1235 | 0.0462 | 0.0773 | 103.51 | 64.79 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0490 | 0.0111 | 0.0379 | 210.81 | 163.21 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2484 | 0.1046 | 0.1438 | 55.63 | 32.20 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0991 | 0.0456 | 0.0535 | 149.47 | 80.72 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2121 | 0.0980 | 0.1141 | 70.13 | 37.72 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4929 | 0.2226 | 0.2704 | 29.59 | 16.23 |
| `apertus` | `apertus` | 8 | 76 | 0.5500 | 0.3054 | 0.2447 | 32.69 | 14.54 |

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
