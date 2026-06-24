# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 15 | Keep notices until replaced. |
| Explicit upstream model ports | 39 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 40 | Safe area for normal refactors. |

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
`.build/benchmarks/test-all-architectures-2026-06-24-independent-lora-data.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0655 | 0.0305 | 0.0350 | 228.81 | 122.18 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1015 | 0.0384 | 0.0632 | 126.68 | 78.78 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1963 | 0.0794 | 0.1169 | 68.41 | 40.75 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3090 | 0.1271 | 0.1819 | 43.98 | 25.89 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0898 | 0.0334 | 0.0564 | 141.87 | 89.10 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2730 | 0.1058 | 0.1672 | 47.84 | 29.31 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0448 | 0.0184 | 0.0264 | 303.07 | 178.76 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0734 | 0.0190 | 0.0544 | 128.72 | 95.43 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1537 | 0.0522 | 0.1016 | 78.78 | 52.04 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2900 | 0.1229 | 0.1672 | 47.85 | 27.58 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.2473 | 0.0992 | 0.1482 | 53.99 | 32.34 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2305 | 0.0869 | 0.1437 | 55.69 | 34.70 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2269 | 0.0812 | 0.1457 | 54.90 | 35.26 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.7225 | 0.5191 | 0.2034 | 39.33 | 11.07 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0930 | 0.0302 | 0.0628 | 127.47 | 86.06 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1286 | 0.0396 | 0.0890 | 89.91 | 62.21 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1418 | 0.0545 | 0.0873 | 91.60 | 56.42 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1293 | 0.0423 | 0.0870 | 91.93 | 61.85 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1450 | 0.0702 | 0.0748 | 106.92 | 55.15 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3029 | 0.1207 | 0.1822 | 43.91 | 26.41 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2418 | 0.1951 | 0.0468 | 171.08 | 33.08 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2443 | 0.1978 | 0.0465 | 172.06 | 32.75 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3970 | 0.2906 | 0.1064 | 75.19 | 20.15 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4720 | 0.3241 | 0.1479 | 54.08 | 16.95 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2308 | 0.1409 | 0.0898 | 89.05 | 34.67 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.3024 | 0.1707 | 0.1318 | 60.72 | 26.45 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1491 | 0.0611 | 0.0880 | 90.87 | 53.65 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0463 | 0.0174 | 0.0289 | 276.98 | 172.92 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4509 | 0.1841 | 0.2668 | 29.98 | 17.74 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0766 | 0.0159 | 0.0607 | 131.84 | 104.42 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1841 | 0.1019 | 0.0822 | 97.31 | 43.46 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0943 | 0.0395 | 0.0548 | 145.87 | 84.80 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0609 | 0.0146 | 0.0464 | 172.59 | 131.28 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1151 | 0.0384 | 0.0768 | 104.19 | 69.48 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.7203 | 0.4258 | 0.2945 | 27.16 | 11.11 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2271 | 0.0870 | 0.1401 | 57.11 | 35.23 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2991 | 0.1121 | 0.1870 | 42.79 | 26.75 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2787 | 0.1059 | 0.1728 | 46.30 | 28.71 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3631 | 0.1396 | 0.2235 | 35.79 | 22.03 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1220 | 0.0443 | 0.0777 | 102.94 | 65.59 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0486 | 0.0103 | 0.0384 | 208.59 | 164.53 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2276 | 0.0836 | 0.1439 | 55.58 | 35.15 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0987 | 0.0472 | 0.0515 | 155.41 | 81.04 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2113 | 0.0975 | 0.1139 | 70.26 | 37.86 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4791 | 0.2121 | 0.2671 | 29.96 | 16.70 |
| `apertus` | `apertus` | 8 | 76 | 0.5520 | 0.3080 | 0.2440 | 32.78 | 14.49 |

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
