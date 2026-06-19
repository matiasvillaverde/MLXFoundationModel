# MLXFoundationModel

Run local MLX models from Swift with Apple's Foundation Models interface.

This package adapts open-source MLX language models to
`LanguageModelSession`. Models run locally on Apple silicon.

> Alpha. The Foundation Models provider path requires Xcode 27 and an OS 27 SDK.
> The direct MLX runtime, tests, and playground also build on current SDKs.

## Requirements

- Apple Silicon Mac for real MLX inference.
- Swift 6.
- Xcode 27 beta and an OS 27 SDK for `LanguageModelSession(model:)`.

## Products

- `MLXFoundationModel`: Foundation Models bridge.
- `MLXLocalModels`: direct MLX generation.
- `MLXFoundationModelExamples`: shared example requests.
- `FoundationModelsPlayground`: runnable examples.

## Usage

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

Build the provider path:

```sh
swift build -Xswiftc -DFOUNDATION_MODELS_PROVIDER_API
```

## Playground

Download the small test model:

```sh
MLX_ASSUME_YES=1 MLX_MODEL_FILTER=smoke make download-test-models
```

Run the examples:

```sh
MLX_FOUNDATION_MODEL_PATH=.models/Qwen3-0.6B-4bit \
MLX_FOUNDATION_MODEL_ID=qwen3-0.6b-4bit \
swift run FoundationModelsPlayground
```

Run one:

```sh
MLX_FOUNDATION_MODEL_PATH=.models/Qwen3-0.6B-4bit \
MLX_FOUNDATION_MODEL_ID=qwen3-0.6b-4bit \
swift run FoundationModelsPlayground --example apple-finite-choice-guided-generation
```

Example output:

```text
=== Apple finite-choice guided generation ===
apple
tokens prompt=31 generated=1 total=32
```

## Features

- Local text generation with MLX models.
- Streaming responses.
- Tool definitions and tool-call parsing.
- Structured output with token-level constrained decoding.
- JSON Schema, JSON, EBNF, regex, and finite-choice constraints.
- Provider compatibility tests behind `FOUNDATION_MODELS_PROVIDER_API`.

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
```

Models download into ignored `.models/` by default. Set `MLX_TEST_MODELS_DIR`
to reuse an existing model directory.
