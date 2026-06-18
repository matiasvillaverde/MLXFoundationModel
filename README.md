# MLXFoundationModel

Use open-source MLX language models behind Apple's Foundation Models-style APIs.

This package is intentionally shaped after
[`ClaudeForFoundationModels`](https://github.com/anthropics/ClaudeForFoundationModels):
there is a low-level local model target with no Foundation Models dependency
(`MLXLocalModels`) and a bridge target (`MLXFoundationModel`) that exposes
an `MLXLanguageModel` entry point.

> Alpha. Apple's server-side/custom-provider Foundation Models APIs are beta and
> require the OS 27 / Xcode 27 SDKs. The package builds on current SDKs with the
> Foundation Models conformance compiled out, so the copied MLX runtime and
> request-rendering tests can keep moving before the beta SDK is installed.

## Requirements

- Apple Silicon Mac for real MLX inference.
- Swift 6.
- For Apple `LanguageModelSession` integration: Xcode 27 beta and an OS 27 SDK
  where `FoundationModels` exposes provider APIs.

## Package Products

- `MLXFoundationModel`: public bridge surface.
- `MLXLocalModels`: direct local MLX generation session for lower-level use.

## Quick Start

```swift
import Foundation
import FoundationModels
import MLXFoundationModel

let model = MLXLanguageModel(
    model: MLXModel(
        id: "mlx-community/Qwen3-4B-4bit",
        location: URL(fileURLWithPath: "/path/to/downloaded/model"),
        promptStyle: .chatML
    )
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Write one sentence about local AI.")
print(response.content)
```

On current SDKs, `MLXLanguageModel` can still be configured and its prompt
renderer can be tested. The `LanguageModel` conformance is gated behind
`FOUNDATION_MODELS_PROVIDER_API` because current public SDKs can import
`FoundationModels` without exposing the provider protocols used by Anthropic's
OS 27 bridge.

With an SDK that exposes those provider protocols, compile the package with:

```sh
swift build -Xswiftc -DFOUNDATION_MODELS_PROVIDER_API
```

## Current Scope

- Text generation from local MLX models.
- Streaming text deltas into Foundation Models executor channels when available.
- Prompt rendering for plain and ChatML-style instruction models.
- Tool definitions rendered into the prompt.
- Structured response constraints rendered into prompts for schema-guided JSON.
- Tool-call extraction through a best-effort JSON parser plus opt-in real-model
  tests for tool-router prompts.
- Foundation Models session overload compatibility tests under
  `FOUNDATION_MODELS_PROVIDER_API`; native `LanguageModelSession(model:)`
  execution requires a host OS where Apple's provider APIs are available.

## Guided Generation

Apple `@Generable` relies on guided generation. For local MLX models, this
package maps Foundation Models schemas into prompt-level JSON constraints.
Token-level grammar/logit masking is still a deeper runtime project, so the
real-model suite validates that supported instruction models follow the rendered
constraints.

## Development

```sh
make lint
make build
make test
```

Real-model tests are opt-in and run serially because MLX uses the GPU:

```sh
MLX_ASSUME_YES=1 MLX_MODEL_FILTER=smoke make download-test-models
make test-real-models

MLX_ASSUME_YES=1 make download-main-models
make test-main-architectures

make test-all-architectures
```

Models download into ignored `.models/` by default. Set `MLX_TEST_MODELS_DIR`
to reuse an existing model directory, `MLX_MODEL_FILTER` to download a subset,
`MLX_REAL_MODEL_SCOPE=main` to require representative main architecture models,
and `MLX_REAL_MODEL_SCOPE=all` to require every downloadable catalog model.
