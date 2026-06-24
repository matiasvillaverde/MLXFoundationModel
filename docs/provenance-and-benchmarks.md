# Provenance and Benchmarks

Last updated: 2026-06-24

## Provenance

This repository is MIT licensed. Some files keep upstream MIT notices from
Apple MLX projects or note ports from MLX model implementations. Keep those
notices unless the file is replaced by an independent implementation.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 27 | Keep notices until replaced. |
| Explicit MLX model ports | 40 | Keep port notes until replaced. |
| Project-authored or no upstream marker | 27 | Safe area for normal refactors. |

No upstream notices were removed in this pass.

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
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0671 | 0.0318 | 0.0353 | 226.46 | 119.20 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1029 | 0.0399 | 0.0631 | 126.84 | 77.73 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1906 | 0.0737 | 0.1170 | 68.38 | 41.96 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.3156 | 0.1337 | 0.1819 | 43.98 | 25.35 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0914 | 0.0347 | 0.0567 | 141.22 | 87.55 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2669 | 0.0995 | 0.1674 | 47.78 | 29.97 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0465 | 0.0185 | 0.0280 | 285.24 | 172.03 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0698 | 0.0188 | 0.0510 | 137.38 | 100.30 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1646 | 0.0604 | 0.1042 | 76.80 | 48.60 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.3006 | 0.1318 | 0.1688 | 47.39 | 26.61 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.3736 | 0.2174 | 0.1561 | 51.23 | 21.42 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2309 | 0.0863 | 0.1446 | 55.34 | 34.65 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2235 | 0.0772 | 0.1463 | 54.68 | 35.80 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 1.2329 | 0.8590 | 0.3740 | 21.39 | 6.49 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0972 | 0.0361 | 0.0611 | 130.87 | 82.29 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1321 | 0.0420 | 0.0901 | 88.81 | 60.58 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1564 | 0.0698 | 0.0866 | 92.35 | 51.15 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1361 | 0.0476 | 0.0885 | 90.35 | 58.77 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1243 | 0.0490 | 0.0754 | 106.16 | 64.35 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3017 | 0.1177 | 0.1839 | 43.50 | 26.52 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2415 | 0.1954 | 0.0461 | 173.57 | 33.13 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2379 | 0.1917 | 0.0462 | 173.20 | 33.62 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3797 | 0.2743 | 0.1054 | 75.89 | 21.07 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4883 | 0.3334 | 0.1549 | 51.65 | 16.38 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2290 | 0.1380 | 0.0909 | 87.98 | 34.94 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.3132 | 0.1813 | 0.1319 | 60.66 | 25.54 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1493 | 0.0614 | 0.0879 | 91.05 | 53.58 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0476 | 0.0186 | 0.0290 | 276.29 | 168.19 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4921 | 0.2021 | 0.2900 | 27.59 | 16.26 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0737 | 0.0157 | 0.0581 | 137.72 | 108.48 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1676 | 0.0892 | 0.0785 | 101.96 | 47.73 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0938 | 0.0393 | 0.0545 | 146.68 | 85.25 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0618 | 0.0152 | 0.0466 | 171.84 | 129.52 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1095 | 0.0339 | 0.0756 | 105.84 | 73.07 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.8339 | 0.5243 | 0.3096 | 25.84 | 9.59 |
| `deepseek_v3` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2697 | 0.1297 | 0.1401 | 57.11 | 29.66 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.3016 | 0.1152 | 0.1864 | 42.92 | 26.53 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2788 | 0.1057 | 0.1731 | 46.22 | 28.69 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3714 | 0.1487 | 0.2227 | 35.93 | 21.54 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1225 | 0.0464 | 0.0761 | 105.19 | 65.32 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0497 | 0.0112 | 0.0385 | 207.84 | 161.03 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2542 | 0.0909 | 0.1632 | 49.01 | 31.48 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.1055 | 0.0461 | 0.0594 | 134.69 | 75.86 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2071 | 0.0921 | 0.1149 | 69.60 | 38.63 |
| `olmo3` | `olmo3` | 8 | 85 | 0.5012 | 0.2338 | 0.2674 | 29.92 | 15.96 |
| `apertus` | `apertus` | 8 | 76 | 0.6311 | 0.3866 | 0.2445 | 32.72 | 12.68 |

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
