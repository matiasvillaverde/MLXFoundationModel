# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 16 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 39 | Safe area for normal refactors. |

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
| `Sources/MLXLocalModels/Common/Module+Extensions.swift` | Logical parameter counting for dense and quantized modules. |
| `Sources/MLXLocalModels/Common/RoPEApplication.swift` | Scalar and batch RoPE offset selection for KV caches. |
| `Sources/MLXLocalModels/Common/Tokenizer.swift` | Tokenizer loading, tokenizer-class rewriting, replacement registry, and streaming detokenization. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-tokenizer.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0646 | 0.0296 | 0.0349 | 229.00 | 123.90 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1000 | 0.0366 | 0.0633 | 126.34 | 80.04 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1882 | 0.0718 | 0.1163 | 68.77 | 42.51 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2927 | 0.1120 | 0.1807 | 44.27 | 27.33 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0907 | 0.0342 | 0.0566 | 141.41 | 88.18 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2716 | 0.1045 | 0.1671 | 47.88 | 29.46 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0463 | 0.0192 | 0.0272 | 294.51 | 172.60 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0723 | 0.0182 | 0.0541 | 129.28 | 96.79 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1557 | 0.0527 | 0.1030 | 77.67 | 51.39 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3035 | 0.1345 | 0.1689 | 47.36 | 26.36 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2484 | 0.1004 | 0.1480 | 54.05 | 32.20 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2287 | 0.0853 | 0.1434 | 55.80 | 34.99 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2302 | 0.0838 | 0.1464 | 54.63 | 34.75 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.8451 | 0.6405 | 0.2046 | 39.10 | 9.47 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0903 | 0.0265 | 0.0638 | 125.45 | 88.61 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1310 | 0.0403 | 0.0907 | 88.25 | 61.07 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1661 | 0.0800 | 0.0862 | 92.84 | 48.15 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1418 | 0.0521 | 0.0897 | 89.20 | 56.44 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1359 | 0.0605 | 0.0754 | 106.17 | 58.87 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3116 | 0.1285 | 0.1831 | 43.69 | 25.67 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2452 | 0.1986 | 0.0466 | 171.58 | 32.63 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2430 | 0.1963 | 0.0466 | 171.57 | 32.93 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3939 | 0.2866 | 0.1073 | 74.57 | 20.31 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4770 | 0.3295 | 0.1475 | 54.22 | 16.77 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2058 | 0.1159 | 0.0899 | 88.98 | 38.88 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2828 | 0.1525 | 0.1303 | 61.41 | 28.29 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1451 | 0.0568 | 0.0883 | 90.62 | 55.13 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0462 | 0.0176 | 0.0285 | 280.34 | 173.35 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4707 | 0.1930 | 0.2777 | 28.81 | 17.00 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0767 | 0.0157 | 0.0610 | 131.16 | 104.26 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1850 | 0.0936 | 0.0914 | 87.48 | 43.23 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0917 | 0.0382 | 0.0535 | 149.47 | 87.23 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0655 | 0.0146 | 0.0509 | 157.15 | 122.12 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1151 | 0.0383 | 0.0768 | 104.16 | 69.52 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.6867 | 0.3920 | 0.2948 | 27.14 | 11.65 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2170 | 0.0771 | 0.1399 | 57.19 | 36.86 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2813 | 0.0970 | 0.1842 | 43.42 | 28.44 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2790 | 0.1061 | 0.1729 | 46.27 | 28.67 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3665 | 0.1447 | 0.2218 | 36.07 | 21.83 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1250 | 0.0481 | 0.0769 | 104.07 | 64.01 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0506 | 0.0104 | 0.0401 | 199.47 | 158.25 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2201 | 0.0760 | 0.1441 | 55.51 | 36.35 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.1017 | 0.0470 | 0.0547 | 146.18 | 78.67 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.1944 | 0.0805 | 0.1140 | 70.20 | 41.15 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4905 | 0.2222 | 0.2684 | 29.81 | 16.31 |
| `apertus` | `apertus` | 8 | 76 | 0.5561 | 0.3105 | 0.2456 | 32.57 | 14.39 |

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
