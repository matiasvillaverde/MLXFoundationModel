# MLX Observability Usage

`MLXLocalModels` exposes a central observability registry through
`MLXObservability`. It is process-global by design so the Foundation Models
bridge, scalar generation, continuous batching, prompt cache, memory guard, and
diagnostic paths all feed the same snapshot and OSLog subsystem.

## Configure

```swift
MLXObservability.configure(
    MLXObservabilityConfiguration(
        osLogEnabled: true,
        signpostsEnabled: true,
        minimumLogSeverity: .info
    )
)
```

Attach a sink when exporting events to a benchmark harness or test:

```swift
final class BenchmarkSink: MLXObservabilitySink {
    func record(_ event: MLXObservabilityEvent) {
        // Export counters, gauges, histograms, spans, and diagnostic logs.
    }

    func recordRequest(_ summary: MLXRequestSummary) {
        // Export per-request tokens, timings, cache reuse, and stop reason.
    }
}
```

## Inspect

```swift
let snapshot = MLXObservability.snapshot()
let requests = snapshot.recentRequests
let counters = snapshot.counters
let histograms = snapshot.histograms
```

Request summaries are redacted. Prompt text, generated token text, and tool
payload content are not exported by the central observability path.

## Instruments

Run the existing profile target to capture OS signposts alongside Time Profiler
or Metal System Trace:

```sh
make profile-real-model
MLX_PROFILE_TEMPLATE='Metal System Trace' make profile-real-model
```

The most useful signpost spans are:

- `model.load`
- `admission.wait`
- `generation`
- `continuous_batch`
- `request.render`
- `stream.translate`

## Real-Model Runs

The real-model runner is serialized and defaults to the ignored `.models`
symlink or `MLX_TEST_MODELS_DIR`.

```sh
make test-real-models
make test-main-architectures
make test-all-architectures
```

On 32 GB hosts, keep the default runner behavior: it runs one model at a time,
uses short generations by default, and skips oversized models unless
`MLX_ALLOW_OVERSIZED_MODELS=1` is set.

The real-model and profiling scripts fail fast when the model volume is not
responsive. Override the default 10-second storage preflight with
`MLX_MODEL_STORAGE_TIMEOUT_SECONDS` when testing slow external disks.
