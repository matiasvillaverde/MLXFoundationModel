# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 25 | Keep notices until replaced. |
| Explicit upstream model ports | 41 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 28 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |

## Code Change

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

These rows come from `BENCH` lines printed by the real-model test runner. They
are short 8-token decode checks, so treat them as a regression snapshot rather
than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0669 | 0.0317 | 0.0352 | 227.23 | 119.51 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1011 | 0.0380 | 0.0632 | 126.68 | 79.11 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1894 | 0.0725 | 0.1168 | 68.47 | 42.24 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3055 | 0.1235 | 0.1821 | 43.94 | 26.18 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0909 | 0.0346 | 0.0562 | 142.24 | 88.02 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2658 | 0.0985 | 0.1673 | 47.82 | 30.10 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0457 | 0.0190 | 0.0267 | 299.34 | 175.09 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0718 | 0.0180 | 0.0537 | 130.28 | 97.52 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1539 | 0.0524 | 0.1016 | 78.78 | 51.97 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2916 | 0.1226 | 0.1689 | 47.36 | 27.44 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.3579 | 0.2103 | 0.1476 | 54.18 | 22.35 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2323 | 0.0880 | 0.1443 | 55.43 | 34.44 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2242 | 0.0780 | 0.1462 | 54.72 | 35.68 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.8368 | 0.6285 | 0.2083 | 38.40 | 9.56 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0992 | 0.0374 | 0.0618 | 129.53 | 80.68 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1307 | 0.0402 | 0.0904 | 88.46 | 61.22 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1703 | 0.0842 | 0.0862 | 92.84 | 46.96 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1354 | 0.0475 | 0.0879 | 91.02 | 59.10 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1434 | 0.0677 | 0.0757 | 105.70 | 55.80 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3038 | 0.1202 | 0.1837 | 43.56 | 26.33 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2390 | 0.1931 | 0.0460 | 174.07 | 33.47 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2409 | 0.1946 | 0.0463 | 172.71 | 33.20 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3941 | 0.2810 | 0.1131 | 70.74 | 20.30 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4799 | 0.3325 | 0.1475 | 54.25 | 16.67 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2014 | 0.1186 | 0.0828 | 96.62 | 39.72 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2951 | 0.1638 | 0.1313 | 60.93 | 27.11 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1474 | 0.0589 | 0.0886 | 90.31 | 54.26 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0458 | 0.0168 | 0.0289 | 276.49 | 174.81 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4612 | 0.1690 | 0.2923 | 27.37 | 17.34 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0730 | 0.0160 | 0.0571 | 140.20 | 109.52 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1618 | 0.0936 | 0.0682 | 117.36 | 49.45 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0970 | 0.0397 | 0.0573 | 139.54 | 82.48 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0590 | 0.0148 | 0.0443 | 180.73 | 135.52 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1113 | 0.0349 | 0.0764 | 104.77 | 71.89 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.8331 | 0.5338 | 0.2993 | 26.73 | 9.60 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2909 | 0.1521 | 0.1389 | 57.60 | 27.50 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3283 | 0.1416 | 0.1867 | 42.84 | 24.37 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2762 | 0.1042 | 0.1720 | 46.50 | 28.96 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3710 | 0.1481 | 0.2229 | 35.89 | 21.56 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1248 | 0.0482 | 0.0765 | 104.54 | 64.12 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0532 | 0.0112 | 0.0420 | 190.46 | 150.27 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2149 | 0.0719 | 0.1430 | 55.96 | 37.23 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0948 | 0.0443 | 0.0505 | 158.30 | 84.35 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2123 | 0.0987 | 0.1136 | 70.42 | 37.69 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4920 | 0.2242 | 0.2678 | 29.87 | 16.26 |
| `apertus` | `apertus` | 8 | 76 | 0.5498 | 0.3053 | 0.2445 | 32.72 | 14.55 |

## Skipped By Memory Gate

| Model | Reason |
| --- | --- |
| `qwen3-moe-30b-a3b-4bit` | Requires 48 GiB RAM. |
| `mistral-small-24b-2501-4bit` | Requires 48 GiB RAM. |
| `phi-3.5-moe-instruct-4bit` | Requires 48 GiB RAM. |
| `gemma-3n-e2b-it-lm-bf16` | Requires 48 GiB RAM. |
| `gemma-3n-e4b-it-lm-bf16` | Estimated runtime memory about 32 GiB; host budget is 24 GiB. |
| `deepseek-r1-4bit` | Requires 256 GiB RAM. |
| `cohere-command-r-v01-4bit` | Estimated runtime memory about 53 GiB; host budget is 24 GiB. |
| `gpt-oss` | Requires 48 GiB RAM. |
| `qwen3-next` | Requires 64 GiB RAM. |
| `qwen3.5-moe` | Requires 48 GiB RAM. |
