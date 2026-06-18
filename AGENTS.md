# Repository Guidelines

## Purpose
`MLXFoundationModel` is a standalone Swift package that adapts MLX-backed
open-source language models to Apple's Foundation Models provider shape.

## Layout
- `Sources/MLXLocalModels/` contains the copied MLX runtime, model registry,
  token generation, prompt cache, and streaming session code.
- `Sources/MLXFoundationModel/` contains the public bridge model,
  request rendering, event translation, and conditional Foundation Models
  conformance.
- `Tests/MLXFoundationModelTests/` covers bridge behavior that does not
  require loading a real model.

## Commands
- `make lint`
- `make build`
- `make test`
- `make test-real-models`
- `make test-main-architectures`
- `make test-all-architectures`
- `make download-test-models`
- `make download-main-models`
- `make quality`

## Rules
- Keep the package standalone. Do not depend on the Patagonia/Think app packages.
- Use SwiftTesting for new tests.
- Use the local Makefile for verification.
- Keep real-model tests opt-in. Default tests must not download model weights.
- Keep model weights out of git. Use `scripts/download-test-models.sh` to
  populate ignored `.models/` or set `MLX_TEST_MODELS_DIR`.
- Use `MLX_REAL_MODEL_SCOPE=main` or `make test-main-architectures` when the
  task needs representative main architecture coverage without downloading the
  whole catalog.
- Files under `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`
  are copied MLX runtime internals and are intentionally excluded from strict lint.
