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
The final request-summary event includes token counts, throughput, cache reuse,
KV cache size, stop reason, grammar kind, and generation controls.

## Core Metrics

Request summaries update these throughput histograms:

- `generation.prompt_tokens_per_second`
- `generation.generation_tokens_per_second`
- `generation.total_tokens_per_second`

The older `generation.tokens_per_second` histogram remains as an alias for
generation-token throughput.

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
For each selected model, the runner verifies generation, sampling/logits
controls, session-style requests when supported, and token-level grammar
constraints.

The real-model and profiling scripts fail fast when the model volume is not
responsive. Override the default 10-second storage preflight with
`MLX_MODEL_STORAGE_TIMEOUT_SECONDS` when testing slow external disks.

Real-model runs print `BENCH` lines with prompt, decode, generated-token e2e,
and total-token throughput: `prompt_tps`, `decode_tps`, `e2e_tps`, and
`total_tps`. They also print matching `BENCH_JSON` lines with stable
machine-readable keys for benchmark comparisons. Stress runs print `STRESS`
and `STRESS_JSON` lines.

The serialized runner also writes its model labels, skip/pass/fail lines, and
Swift test output to `.build/benchmarks/real-models-<timestamp>.log` by default.
It writes a compact parsed summary to
`.build/benchmarks/real-models-<timestamp>-summary.json`. Override the log with
`MLX_REAL_MODEL_BENCHMARK_LOG`, the summary with
`MLX_REAL_MODEL_BENCHMARK_SUMMARY`, or the directory with
`MLX_REAL_MODEL_BENCHMARK_DIR`.

Compare two summary files with:

```sh
make compare-benchmarks BASELINE=.build/benchmarks/old-summary.json \
  CURRENT=.build/benchmarks/new-summary.json
```

`COMPARE_MIN_RATIO` controls the minimum current/baseline throughput ratio for
`decode_tps`, `total_tps`, `prompt_tps`, and `e2e_tps`.
