# MLXFoundationModel

Local MLX models through Apple APIs.

`MLXFoundationModel` is a Swift package for running open-source MLX language
models locally and using them through the same shape as Apple's Foundation
Models APIs: streaming text, tool calls, structured output, and
`LanguageModelSession` when the provider API is available.

Nothing is uploaded. Real-model tests are part of the repo. The goal is
practical: app code can use Apple APIs while model support comes from open MLX
weights.

> Alpha. Direct MLX generation works on current SDKs. The
> `LanguageModelSession(model:)` provider path requires Xcode 27 and an OS 27
> SDK.

## Try It

```sh
make demo
```

This downloads Qwen3 0.6B 4-bit into ignored `.models/` and runs the playground.

Useful variants:

```sh
make demo DEMO_EXAMPLE=apple-finite-choice-guided-generation
swift run FoundationModelsPlayground --list-examples
```

## Test

```sh
make test
```

Real-model checks are opt-in:

```sh
make test-demo-model          # downloads and tests one small model
make download-main-models
make test-main-architectures  # serialized across representative models
```

Use `MLX_TEST_MODELS_DIR=/path/to/models` to reuse shared model storage.

## Add Package

```swift
.package(
    url: "https://github.com/matiasvillaverde/MLXFoundationModel.git",
    branch: "main"
)
```

```swift
.product(name: "MLXFoundationModel", package: "MLXFoundationModel")
```

## Use LanguageModelSession

```swift
import Foundation
import FoundationModels
import MLXFoundationModel

let model = try MLXLanguageModel(
    id: "qwen3-0.6b-4bit",
    location: URL(fileURLWithPath: ".models/Qwen3-0.6B-4bit")
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Write one sentence about local inference.")
print(response.content)
```

Build that path with:

```sh
swift build -Xswiftc -DFOUNDATION_MODELS_PROVIDER_API
make test-provider
```

## Works Today

- MLX-backed text generation on Apple silicon.
- Request rendering for the Foundation Models shape.
- Streaming text translation.
- Tool-call parsing and schema normalization.
- JSON Schema, JSON, EBNF, regex, and finite-choice constraints.
- Prompt caching, memory guards, model pooling, and real-model timing.
- Metrics, logs, and Instruments signposts.

## Layout

- `Sources/MLXFoundationModel`: public bridge.
- `Sources/MLXLocalModels`: MLX runtime, model registry, generation, caching.
- `Examples/FoundationModelsPlayground`: runnable examples.
- `Tests/MLXFoundationModelTests`: unit tests without real weights.
- `Tests/MLXRealModelTests`: opt-in tests with downloaded models.
- `docs/observability-usage.md`: metrics and logging.
- `docs/apple-profiling.md`: Instruments workflows.

## Commands

```sh
make help
make build
make test
make demo
make test-demo-model
make quality
```

Model downloads are ignored by git. The default location is `.models/`.

## License

MIT. See `NOTICE.md` for third-party attribution.
