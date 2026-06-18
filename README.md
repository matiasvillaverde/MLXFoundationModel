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
- Tool-call extraction is scaffolded as a best-effort JSON parser; true tool
  calling parity requires dialect-specific constrained output work.

## Guided Generation

Apple `@Generable` relies on guided generation. For local MLX models, this
requires constrained decoding or schema-aware logit masking. This package keeps
the bridge points in place but does not yet claim full `@Generable` parity.

## Development

```sh
make lint
make build
make test
```
