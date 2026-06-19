# MLXFoundationModel

Use open-source MLX language models with Apple's Foundation Models interface.

MLXFoundationModel is a Swift package for apps that want Apple's
`LanguageModelSession` interface, but with local models that run through MLX.
The package keeps the MLX runtime in `MLXLocalModels` and exposes the
Foundation Models bridge from `MLXFoundationModel`.

> Alpha. Apple's custom-provider APIs require the OS 27 / Xcode 27 SDKs. The
> package also builds on current SDKs with the provider conformance disabled, so
> the local runtime, playground, and tests can run before those APIs are
> available on the host.

## Requirements

- Apple Silicon Mac for real MLX inference.
- Swift 6.
- Xcode 27 beta and an OS 27 SDK for `LanguageModelSession(model:)`
  integration.

## Products

- `MLXFoundationModel`: public bridge surface.
- `MLXLocalModels`: direct local MLX generation session.
- `MLXFoundationModelExamples`: shared examples for tests and the playground.
- `FoundationModelsPlayground`: executable playground for a local model.

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

Build the provider path with an SDK that exposes Apple's custom-provider
protocols:

```sh
swift build -Xswiftc -DFOUNDATION_MODELS_PROVIDER_API
```

## Playground Examples

The playground runs the same examples as the opt-in real-model tests:

- streaming chat
- structured Trip Planner output
- tool-call JSON
- finite choices: `apple`, `pear`, or `banana`
- structured content tags

Download the smallest smoke-test model:

```sh
MLX_ASSUME_YES=1 MLX_MODEL_FILTER=smoke make download-test-models
```

Run all examples:

```sh
MLX_FOUNDATION_MODEL_PATH=.models/Qwen3-0.6B-4bit \
MLX_FOUNDATION_MODEL_ID=qwen3-0.6b-4bit \
swift run FoundationModelsPlayground
```

Run one example:

```sh
MLX_FOUNDATION_MODEL_PATH=.models/Qwen3-0.6B-4bit \
MLX_FOUNDATION_MODEL_ID=qwen3-0.6b-4bit \
swift run FoundationModelsPlayground --example apple-finite-choice-guided-generation
```

The playground prints streamed model output and token usage:

```text
=== Apple finite-choice guided generation ===
apple
tokens prompt=31 generated=1 total=32
```

On OS 27 with provider APIs available, the playground can exercise the
`LanguageModelSession(model:)` path. On older hosts it uses the direct MLX
session path with the same prompts, tools, schemas, and constraints.

## Current Scope

- Text generation from local MLX models.
- Streaming text deltas into Foundation Models executor channels when available.
- Prompt rendering for plain and ChatML-style instruction models.
- Tool definitions rendered into the prompt.
- Structured response constraints mapped to token-level constrained decoding.
- Tool-call extraction through a best-effort JSON parser.
- Foundation Models session overload compatibility tests under
  `FOUNDATION_MODELS_PROVIDER_API`.

## Guided Generation

Guided generation is enforced during decoding, not by prompt text alone.
Foundation Models schemas are mapped to XGrammar-backed token masks before each
token is sampled. Supported constraints include JSON Schema, builtin JSON, EBNF,
regex, and finite choices such as `apple | pear | banana`.

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
