# Observability and Profiling Research

This note captures research on how adjacent local and server LLM projects expose
tracing and metrics, then maps Apple's profiling, logging, and MetricKit tools to
`MLXFoundationModel`.

The target environment for the Apple-specific section is Xcode 27 on macOS 27.

## Executive Summary

`MLXFoundationModel` should treat observability as four layers:

1. In-process metrics that are always cheap enough to collect.
2. Optional request traces for detailed local diagnosis.
3. Apple-native logging and signposts for Instruments.
4. Host-app production reporting through MetricKit or other exporters.

The package should not make Prometheus, OpenTelemetry, or MetricKit hard
dependencies. It is a Swift package embedded in host apps, not a long-running
server by itself. The right shape is a typed metrics and trace sink that hosts
can adapt to Prometheus, OpenTelemetry, MetricKit, OSLog, JSON, or dashboards.

The highest-value Apple integration is `OSSignposter` intervals around the model
request lifecycle. Signposts make Time Profiler, System Trace, Metal System
Trace, Logging, and the Xcode 27 Foundation Models instrument easier to align.
MetricKit is useful for coarse production health, but it is not a replacement
for per-generation token, cache, and latency metrics.

## North Star

Observability should be a first-class feature of this package, not an afterthought
bolted onto generation.

The target experience:

1. Every generation produces a compact, privacy-safe request summary.
2. Every major runtime phase can be traced with stable IDs and timings.
3. A host app can expose the same data as logs, JSON, Prometheus, OpenTelemetry,
   MetricKit signpost reports, or Instruments traces without rewriting runtime
   code.
4. A developer can answer performance questions with a repeatable workflow:
   run a real model, capture metrics, record a short trace, compare with a
   baseline, and know what knob or subsystem to inspect next.
5. The package never logs prompt text, response text, tool arguments, or user
   data unless a host explicitly enables a local debug mode.

The end result should feel closer to an inference lab than a black box: the
runtime should explain where time, memory, tokens, cache hits, and failures went.

## Other LLM Platforms

### vLLM

vLLM has the richest production observability model.

It exposes Prometheus-compatible metrics at `/metrics`, uses a `vllm:` metric
prefix, and separates server-level metrics from request-level histograms:
<https://docs.vllm.ai/en/stable/design/metrics/>

Useful pieces to copy:

- Histograms for time to first token, inter-token latency, end-to-end latency,
  queue time, prefill time, decode time, and per-token latency.
- Counters for prompt tokens, generated tokens, cached prompt tokens, request
  outcomes, prefix-cache queries, and prefix-cache hits.
- Gauges for running, waiting, and deferred requests, plus KV cache usage.
- Optional OpenTelemetry tracing for individual requests:
  <https://docs.vllm.ai/en/stable/design/metrics/#tracing>
- Optional detailed traces and sampled KV cache metrics to control overhead:
  <https://docs.vllm.ai/en/stable/api/vllm/config/observability/>

Design lesson: expose a stable, low-cardinality metric vocabulary and keep
expensive traces opt-in.

### llama.cpp

llama.cpp keeps the operational surface lean.

The server exposes opt-in Prometheus metrics with `--metrics`, including prompt
tokens/time, generated tokens/time, processing and deferred request gauges,
context high-water marks, decode-call counts, and busy-slot counts:
<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#metrics>

It also exposes a slots endpoint for per-slot/request monitoring:
<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#slots>

Its C API includes resettable performance structs for prompt evaluation,
generation evaluation, sampling, and reused tokens:
<https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h>

Design lesson: keep a compact local perf model close to the executor, and
separate cumulative totals from resettable interval buckets.

### MLX LM

MLX LM is library-oriented rather than server-observability-oriented.

Generation responses include prompt tokens-per-second, generation
tokens-per-second, peak memory, token counts, and finish information:
<https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py>

Its benchmark script measures time to first token, per-request tokens/sec,
aggregate tokens/sec, and p95-style summaries:
<https://github.com/ml-explore/mlx-lm/blob/main/benchmarks/server_benchmark.py>

Design lesson: every final generation result should carry a useful summary:
prompt tokens, generated tokens, prompt TPS, generation TPS, time to first token,
total latency, and peak memory where available.

### oMLX

oMLX is the closest reference for Apple Silicon server/runtime behavior.

It maintains aggregate server metrics for prompt tokens, completion tokens,
cached tokens, request counts, average prefill TPS, average generation TPS, and
cache efficiency:
<https://github.com/jundot/omlx/blob/main/omlx/server_metrics.py>

It tracks scheduler state such as waiting, prefilling, running, processed
requests, prompt tokens, completion tokens, SSD/prefix-cache counters, cache
hits/misses, tokens matched/requested/saved, evictions, SSD hot hits, disk loads,
saves, and errors:
<https://github.com/jundot/omlx/blob/main/omlx/scheduler.py>

It also has Apple-relevant memory monitoring and admin-dashboard data for loaded
models, active requests, waiting requests, model memory, memory pressure, hot
cache, and paged SSD cache:
<https://github.com/jundot/omlx/blob/main/omlx/memory_monitor.py>

An oMLX issue documents the same operator need we have: aggregate inference
metrics are not enough without CPU/GPU usage, model-weight memory, hot KV cache,
runtime/activation memory, macOS memory pressure, swap, and compressor data:
<https://github.com/jundot/omlx/issues/1604>

Design lesson: Apple Silicon inference needs cache and memory-pressure metrics
as first-class signals, not just token throughput.

### OpenTelemetry GenAI

OpenTelemetry's GenAI semantic conventions matter because they define the
vendor-neutral vocabulary that many observability tools now understand:
<https://github.com/open-telemetry/semantic-conventions-genai>

The main OpenTelemetry semantic convention registry has moved GenAI attributes
into that dedicated repository, but the registry still shows the shape of the
standard vocabulary:
<https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/>

Useful concepts to align with:

- Operation name, such as chat or text completion.
- Provider and model identifiers.
- Request parameters such as max tokens, temperature, top-p, top-k, seed, and
  streaming mode.
- Response finish reasons and response model.
- Time to first chunk.
- Input tokens, output tokens, cached input tokens, cache creation tokens, and
  reasoning output tokens.
- Tool definitions, tool call IDs, tool names, and tool results.
- Conversation/session/workflow identifiers.

Design lesson: use OTel-compatible names and concepts where they fit, but keep a
package-native schema as the source of truth. Some OTel attributes, especially
input messages, output messages, system instructions, tool arguments, and tool
results, are sensitive by default and should remain disabled unless the host
opts into content capture.

## Parity Review

The table below maps what mature systems observe to what this package already
has and what it needs next.

| Area | What others observe | Current package state | Target for this repo |
| --- | --- | --- | --- |
| Request lifecycle | vLLM tracks queue, prefill, decode, TTFT, inter-token latency, and end-to-end latency. llama.cpp tracks prompt/eval timings. MLX LM returns prompt/generation TPS. | `ChunkMetrics` has total time, TTFT, prompt processing time, token counts, and generation parameters. | Add phase-level request spans and histograms for render, tokenize, admission, cache lookup, prefill, first token, decode, tool parse, stream translate, and cache save. |
| Scheduler and concurrency | vLLM reports running, waiting, deferred, preemptions, and KV pressure. oMLX reports waiting, prefilling, running, active requests, and queue state. | Continuous batching and admission diagnostics exist internally. | Expose active/queued/prefilling/decoding gauges, queue wait histograms, admission rejection counters, and batch-size/row-count snapshots. |
| Token usage | OTel standardizes input, output, cached input, cache creation, and reasoning tokens. vLLM and llama.cpp expose prompt/generated token counters. | Usage metrics include prompt, generated, total, and prompt-cache reused tokens. Reasoning is visible in stream reducers and diagnostics, but not as a first-class token usage breakdown. | Track input, generated, cached input, reasoning output, tool-call output, and total context tokens consistently. |
| Throughput | vLLM reports prompt and generation throughput. llama.cpp reports prompt and predicted tokens/sec. MLX LM reports prompt TPS, generation TPS, and peak memory. | Logs include generation tokens/sec; final metrics can derive some rates. | Put prompt TPS, generation TPS, total TPS, and accepted-token TPS for speculative/MTP paths into the final request summary. |
| Cache behavior | vLLM tracks prefix-cache queries/hits and optional sampled KV residency. oMLX tracks prefix hits/misses, matched/requested/saved tokens, evictions, SSD hot hits, disk loads, saves, errors, hot-cache evictions/promotions. | Prompt-cache observability already tracks many of these counters internally. | Promote cache counters into public snapshots, add cache lookup/restore/save durations, and sample KV block lifetime/reuse gap metrics when paged KV is enabled. |
| Memory pressure | oMLX emphasizes model weights, hot cache, runtime/activation memory, macOS pressure, swap, and compressor data. MLX LM reports peak memory. MetricKit reports coarse production memory and memory exceptions. | Memory guard diagnostics estimate peaks, limits, and current memory. Model-pool snapshots estimate resident weight bytes. | Add runtime memory snapshot API with model weights, KV/cache bytes, prompt-cache hot bytes, persistent cache bytes, current active memory, limit source, estimated peak, and guard tier. |
| Model loading and residency | llama.cpp exposes load timing in perf context. oMLX exposes loaded models and model memory in admin stats. | Model-pool snapshots expose resident IDs and estimated bytes. Load/unload logs exist. | Add model load/unload spans, model load time histogram, resident-model gauges, pinned/leased/pending unload gauges, and model switch counters. |
| GPU/Metal work | Apple tooling observes Metal command buffers, GPU counters, scheduling, and frame/system traces. vLLM has CUDA/NVTX details on GPU platforms. | `scripts/profile-real-model.sh` can record `Metal System Trace`; no package signposts yet. | Add signposts around CPU phases so Metal traces can be aligned with MLX GPU work. Keep direct GPU internals to Instruments/MLX rather than trying to fake kernel-level metrics. |
| Logs | llama.cpp and MLX LM print human-readable summaries. vLLM logs interval throughput and queue/cache state. Apple recommends structured `Logger`. | Runtime uses `Logger`, but subsystem/category naming is inconsistent and logs are not a complete operational story. | Standardize subsystems/categories, make logs privacy-safe, and emit durable state transitions plus concise request summaries. |
| Traces | vLLM supports OTel traces and optional detailed model execution timing. Apple provides Instruments and signposts. OTel defines GenAI spans/events. | Internal diagnostics are test-friendly but not host-exportable traces. | Add `MLXRequestTrace` and sink protocol; bridge to OS signposts first, then optional OTel. |
| Debuggability | vLLM and oMLX dashboards help explain capacity and cache behavior. Apple Instruments explains CPU/GPU/system causes. | Tests inspect diagnostics, but a developer still needs to know which metrics to read. | Add symptom-based playbooks and a generated per-run observability report bundle. |
| Privacy | OTel warns content attributes can contain PII. Apple logging defaults to privacy protection. | Metrics are mostly numeric today. Some diagnostics include token text for tests. | Keep production observability numeric/structural. Content capture must be explicit, local-only by default, bounded, and redacted/truncated. |

## How We Can Be Better

Matching vLLM's metric list is not enough. This package can be better because it
runs inside Apple's native developer environment and already has domain-specific
knowledge of Foundation Models bridging, prompt rendering, tool parsing,
grammar-constrained decoding, prompt-cache storage, and memory guards.

### Better Than vLLM

vLLM is strong on server metrics, Prometheus, and OTel. This repo can go further
for local Apple inference by making Instruments and signposts part of the core
workflow:

- Every request trace should align with Xcode's Time Profiler, System Trace,
  Metal System Trace, Logging, and Foundation Models instruments.
- The same request ID should appear in metrics snapshots, logs, signposts, real
  model test output, and profiling sidecar JSON.
- Direct MLX and Foundation Models provider paths should share the same
  high-level metrics vocabulary so regressions can be compared across runtimes.
- Memory guard decisions should be visible as structured spans and counters, not
  only thrown errors.

### Better Than llama.cpp

llama.cpp is excellent at simple timing and throughput. This repo can keep that
simplicity while adding richer context:

- Final request summaries should be as easy to read as llama.cpp timings, but
  should include cache reuse, memory estimate, strategy, grammar/tool mode, and
  stop reason.
- Resettable interval buckets should support tests and local profiling without
  requiring a server endpoint.
- Active request snapshots should behave like llama.cpp slots, but with
  Foundation Models concepts: prompt style, provider path, tool mode, grammar
  mode, cache fingerprint, and stream state.

### Better Than MLX LM

MLX LM's generation response stats are useful and low overhead. This repo can
adopt that shape while making it more operational:

- Include prompt TPS, generation TPS, TTFT, total latency, and peak/estimated
  memory in every final chunk or request summary.
- Add per-phase spans so a low TPS value is explainable.
- Include optimization-profile state: continuous batching, speculative decoding,
  dFlash, spec prefill, KV quantization, reasoning budget, and grammar mode.

### Better Than oMLX

oMLX is closest to the Apple Silicon problem space and has strong cache/memory
signals. This repo can be cleaner because it is a package, not a server UI:

- Define typed Swift snapshots instead of ad hoc dictionaries.
- Keep observability independent from any admin dashboard.
- Make host export explicit through protocols.
- Put privacy guarantees in the type model.
- Keep metrics names and cardinality stable enough for long-lived dashboards.

### Better Than Generic OTel

OTel gives interoperability but cannot know MLX-specific internals:

- Use OTel-compatible names for operation, model, token usage, finish reason,
  streaming mode, and tool calls.
- Add MLX-specific fields for cache layout, memory guard tier, prompt renderer,
  generation strategy, paged KV state, and Foundation Models provider mapping.
- Avoid capturing content attributes by default even if OTel supports them.

## Current State in This Repository

The package already has a strong foundation:

- `ChunkMetrics` exposes timing, usage, and generation metadata in stream chunks:
  [LocalGenerationTypes.swift](../Sources/MLXLocalModels/LocalGenerationTypes.swift)
- `MetricsData+ChunkMetrics` calculates prompt-processing time, time to first
  token, token counts, KV cache bytes/entries, prompt-cache reuse, and generation
  parameters:
  [MetricsData+ChunkMetrics.swift](../Sources/MLXLocalModels/MetricsData+ChunkMetrics.swift)
- `MLXGenerationDiagnosticEvent` already records internal events for generation
  parameters, prompt-cache plans, speculative decoding, dFlash planning,
  prefill chunks, cache snapshots, grammar constraints, memory guard decisions,
  admission, batch rows, paged KV blocks, and prompt-cache observability:
  [GenerationContext.swift](../Sources/MLXLocalModels/Common/GenerationContext.swift)
- Prompt-cache observability already tracks hits, misses, tokens requested,
  tokens matched, evictions, SSD hot hits, disk loads, saves, hot-cache
  evictions, promotions, and rate windows:
  [MLXPromptCacheObservability.swift](../Sources/MLXLocalModels/MLXPromptCacheObservability.swift)
- `MLXModelPoolSnapshot` exposes model-pool state, resident models, pinned and
  leased models, pending unloads, and estimated resident weight bytes:
  [MLXModelPoolSnapshot.swift](../Sources/MLXFoundationModel/MLXModelPoolSnapshot.swift)
- `scripts/profile-real-model.sh` already records release-mode Instruments
  traces into `.build/reports/profiles`.

The missing layer is a stable observability API that turns internal diagnostics
into typed counters, gauges, histograms, and trace spans that host apps can
export.

## Apple Tooling Research

### Unified Logging

Apple's Unified Logging stack is the production-safe logging baseline for Swift
apps:

- `Logger`: <https://developer.apple.com/documentation/os/logger>
- Logging overview:
  <https://developer.apple.com/documentation/os/logging>
- Generating log messages:
  <https://developer.apple.com/documentation/os/generating-log-messages-from-your-code>
- WWDC20 "Explore logging in Swift":
  <https://developer.apple.com/videos/play/wwdc2020/10168/>
- WWDC23 "Debug with structured logging":
  <https://developer.apple.com/videos/play/wwdc2023/10226/>

Important guidance:

- Prefer `Logger` and Unified Logging over `print` for diagnostics.
- Use stable subsystems and categories so Xcode, Console.app, `log`, and
  Instruments can filter by component.
- Swift logging interpolation is designed to avoid eager string formatting cost.
- Treat privacy annotations as part of the schema. Strings and objects are
  private by default. Mark only safe values as public.
- Use `debug` for verbose development diagnostics, `info` for routine
  operational detail, `notice` for durable state transitions, and `error` or
  `fault` sparingly.
- Do not log prompts, responses, tool arguments, or user data by default. If a
  host app wants prompt capture for local debugging, make that an explicit debug
  mode and mark the trace files as sensitive.

Recommended categories for this package:

- `generation`
- `model-pool`
- `prompt-cache`
- `memory-guard`
- `provider`
- `tool-parser`
- `grammar`
- `streaming`

The current package already imports `OSLog` and uses `Logger` in several runtime
paths, but subsystem strings such as `MLXSession` should eventually be made
stable and package-scoped, for example `org.mlxfoundationmodel`.

### Signposts

Signposts are the main bridge between code-level instrumentation and
Instruments:

- `OSSignposter`: <https://developer.apple.com/documentation/os/ossignposter>
- Recording performance data:
  <https://developer.apple.com/documentation/os/recording-performance-data>
- WWDC18 "Measuring Performance Using Logging":
  <https://developer.apple.com/videos/play/wwdc2018/405/>
- WWDC26 "Profile, fix, and verify":
  <https://developer.apple.com/videos/play/wwdc2026/268/>

Use intervals for request phases:

- `model.load`
- `model.unload`
- `request.render`
- `tokenize`
- `admission.wait`
- `prompt_cache.lookup`
- `prompt_cache.restore`
- `memory_guard.check`
- `prefill`
- `first_token`
- `decode`
- `grammar.mask`
- `tool.parse`
- `stream.translate`
- `prompt_cache.save`

Avoid per-token signposts by default. They can become too high-volume and make
traces harder to interpret. Prefer aggregate decode intervals, sampled token
events, or counters in the final metrics snapshot.

For overlapping async work, use stable signpost IDs and balanced begin/end
events. Points of Interest signposts are useful for aligning repeated benchmark
runs and Instruments run comparisons.

### Instruments and xctrace

Apple's core profiling loop is:

1. Build and profile a release configuration.
2. Add signposts around the exact workload.
3. Record short, repeatable traces.
4. Compare baseline and candidate runs over the same signposted interval.

Sources:

- Xcode command-line tool reference:
  <https://developer.apple.com/documentation/xcode/xcode-command-line-tool-reference>
- Improving app performance:
  <https://developer.apple.com/documentation/xcode/improving-your-app-s-performance>
- Gathering information about memory use:
  <https://developer.apple.com/documentation/xcode/gathering-information-about-memory-use>
- Analyzing the performance of your Metal app:
  <https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app>
- WWDC26 "Profile, fix, and verify":
  <https://developer.apple.com/videos/play/wwdc2026/268/>

On this Xcode 27 machine, `xcrun xctrace list templates` includes:

- `Foundation Models`
- `Core AI`
- `Core ML`
- `Metal System Trace`
- `Logging`
- `System Trace`
- `Time Profiler`
- `Allocations`
- `Leaks`
- `File Activity`
- `Swift Concurrency`
- `Processor Trace`

Template choice for this package:

- Use `Foundation Models` for the macOS 27 Foundation Models provider path. The
  WWDC26 session says the enhanced Foundation Models instrument can inspect
  sessions, requests, model inferences, instructions, prompts, responses, tool
  calls, time to first token, tokens per second, and total latency, and supports
  any model used with the Foundation Models framework:
  <https://developer.apple.com/videos/play/wwdc2026/243/>
- Use `Time Profiler` for Swift CPU cost: prompt rendering, tokenization, JSON
  extraction, tool parsing, sampling, stream translation, and executor overhead.
- Use `Metal System Trace` for direct MLX GPU behavior: command-buffer gaps,
  CPU/GPU overlap, GPU starvation, and Metal scheduling.
- Use `System Trace` for blocking, thread scheduling, QoS issues, locks, VM
  activity, and cases where wall-clock latency is high but CPU samples are not.
- Use `Allocations`, `Leaks`, and memory graphs for model loading, cache
  retention, prompt-cache payloads, and suspected leaks.
- Use `File Activity` for safetensors loading and persistent prompt-cache I/O.
- Use `Logging` for OSLog and signpost-focused investigations.
- Use `Swift Concurrency` for async stream delivery, actor/executor contention,
  and structured concurrency behavior.
- Use `Processor Trace` only when supported by the hardware and when exact CPU
  control-flow visibility is worth the trace size and cost.

Existing local workflow:

```sh
xcrun xctrace list templates
scripts/profile-real-model.sh
MLX_PROFILE_TEMPLATE='Metal System Trace' scripts/profile-real-model.sh
MLX_PROFILE_TEMPLATE='File Activity' MLX_PROFILE_TIME_LIMIT=15s scripts/profile-real-model.sh
MLX_PROFILE_TEMPLATE='Allocations' scripts/profile-real-model.sh
```

Xcode 27 supports template recording options through:

```sh
xcrun xctrace record --template 'Time Profiler' --show-recording-options
```

Examples observed locally:

- `Time Profiler`, `System Trace`, and `Metal System Trace` expose options for
  context switch sampling, high-frequency sampling, kernel stacks, waiting
  threads, and hang thresholds.
- `Logging` exposes OS log and signpost process-capture options.
- `Core AI` exposes a numeric function suffix consolidation option.
- `Foundation Models` currently reports an empty recording-options object.

### MetricKit

MetricKit is app-level production telemetry. It is useful for real-world health
signals, not detailed per-request LLM telemetry.

Sources:

- MetricKit overview:
  <https://developer.apple.com/documentation/metrickit>
- Monitoring app performance with MetricKit:
  <https://developer.apple.com/documentation/metrickit/monitoring-app-performance-with-metrickit>
- Analyzing app performance with MetricKit:
  <https://developer.apple.com/documentation/metrickit/analyzing-app-performance-with-metrickit>
- Legacy `MXMetricManager`:
  <https://developer.apple.com/documentation/metrickit/mxmetricmanager>
- New `MetricManager`:
  <https://developer.apple.com/documentation/metrickit/metricmanager>
- `MetricReport`:
  <https://developer.apple.com/documentation/metrickit/metricreport>
- `DiagnosticReport`:
  <https://developer.apple.com/documentation/metrickit/diagnosticreport>
- `SignpostIntervalMetric`:
  <https://developer.apple.com/documentation/metrickit/signpostintervalmetric>
- WWDC26 "Meet the new MetricKit":
  <https://developer.apple.com/videos/play/wwdc2026/222/>

Key findings:

- MetricKit collects system-captured app performance and diagnostics from real
  devices, including CPU, memory, network, launch, disk I/O, GPU/display,
  responsiveness, exits, terminations, crashes, hangs, and custom signpost
  intervals.
- The legacy API uses `MXMetricManager.shared` and
  `MXMetricManagerSubscriber`. Daily metric payloads arrive at most once per day.
- On macOS 27 and iOS 27, Apple introduces a Swift-first `MetricManager` with
  async `metricReports` and `diagnosticReports`, `Codable` reports, and state
  reporting support.
- The old `MXMetricManager` and several `MX*` payload types are deprecated on
  macOS 27 in favor of the new API.
- `MetricReport` covers a 24-hour reporting period. `DiagnosticReport` is an
  event-style report for diagnostic events.
- State reporting can segment some metrics by app-defined user-visible states,
  but CPU, memory, network, disk I/O, GPU, app launch, and disk-space metrics
  remain in the interval aggregate rather than state entries.
- Custom signpost interval metrics can capture count and duration data for
  operations like generation or cache restore, but they should be used for
  important sections, not extremely high-cardinality per-token events.

Limits for this package:

- MetricKit reports belong to the host app process and bundle. A Swift package
  cannot independently receive production reports unless the host app integrates
  MetricKit and keeps the manager or subscriber alive.
- MetricKit does not provide real-time request telemetry.
- MetricKit does not know prompt token count, generated token count, cache
  reuse, model ID, grammar mode, tool-parser failures, or MLX-specific cache
  state unless the app records that information elsewhere.
- MetricKit should be treated as coarse production health and regression
  evidence, not as the package's primary metrics layer.

Recommended package stance:

- Provide signpost hooks that host apps can benefit from in Instruments and
  MetricKit.
- Provide typed metrics snapshots and request summaries in package APIs.
- Provide a host-app MetricKit integration example later, but keep MetricKit out
  of the core package dependency surface.

### XCTest Performance Metrics

XCTest can measure performance in repeatable tests:

- Performance tests:
  <https://developer.apple.com/documentation/xctest/performance-tests>
- `XCTClockMetric`: <https://developer.apple.com/documentation/xctest/xctclockmetric>
- `XCTCPUMetric`: <https://developer.apple.com/documentation/xctest/xctcpumetric>
- `XCTMemoryMetric`:
  <https://developer.apple.com/documentation/xctest/xctmemorymetric>
- `XCTOSSignpostMetric`:
  <https://developer.apple.com/documentation/xctest/xctossignpostmetric>

This repository uses Swift Testing, so XCTest metrics are not the primary test
interface today. Still, the model is useful: signpost the operation, run it
repeatedly in release mode, and compare clock, CPU, memory, and signposted
duration across revisions.

## Proposed Metrics Taxonomy

### Request Counters

- `mlx.requests.total`
- `mlx.requests.finished.total`, labeled by finish reason
- `mlx.requests.failed.total`, labeled by error class
- `mlx.tokens.prompt.total`
- `mlx.tokens.generated.total`
- `mlx.tokens.cached_prompt.total`
- `mlx.tool_calls.total`
- `mlx.tool_parser_failures.total`
- `mlx.grammar_rejections.total`
- `mlx.memory_guard.rejections.total`
- `mlx.prompt_cache.queries.total`
- `mlx.prompt_cache.hits.total`
- `mlx.prompt_cache.evictions.total`
- `mlx.prompt_cache.ssd_loads.total`
- `mlx.prompt_cache.ssd_saves.total`

### Runtime Gauges

- `mlx.model_pool.resident_models`
- `mlx.model_pool.resident_weight_bytes`
- `mlx.model_pool.loading_models`
- `mlx.requests.active`
- `mlx.requests.queued`
- `mlx.cache.kv_bytes`
- `mlx.cache.kv_entries`
- `mlx.prompt_cache.hot_bytes`
- `mlx.prompt_cache.ssd_bytes`
- `mlx.memory.current_bytes`
- `mlx.memory.limit_bytes`
- `mlx.memory.estimated_peak_bytes`

### Histograms

- `mlx.request.latency_seconds`
- `mlx.request.time_to_first_token_seconds`
- `mlx.request.queue_seconds`
- `mlx.request.prefill_seconds`
- `mlx.request.decode_seconds`
- `mlx.request.inter_token_seconds`
- `mlx.request.prompt_tokens`
- `mlx.request.generated_tokens`
- `mlx.request.cached_prompt_tokens`
- `mlx.model.load_seconds`
- `mlx.prompt_cache.lookup_seconds`
- `mlx.prompt_cache.restore_seconds`
- `mlx.prompt_cache.save_seconds`
- `mlx.grammar.mask_seconds`
- `mlx.tool_parse_seconds`

### Request Trace Spans

- `model.load`
- `request.render`
- `tokenize`
- `admission`
- `prompt_cache.lookup`
- `memory_guard`
- `prefill`
- `decode`
- `stream`
- `tool_parse`
- `grammar`
- `prompt_cache.save`
- `model.unload`

Keep span attributes low-cardinality and privacy-safe:

- `model.id` or a stable non-sensitive alias
- `runtime.kind`
- `strategy`
- `prompt_tokens`
- `generated_tokens`
- `cached_tokens`
- `finish_reason`
- `tool_call_count`
- `grammar_kind`

Do not use prompt text, response text, tool arguments, file paths, or arbitrary
errors as metric labels.

## Debugging Playbooks

Observability becomes useful when it points to the next action. These playbooks
define the expected workflow for common failures and regressions.

### High Time to First Token

Signals to inspect:

- `mlx.request.time_to_first_token_seconds`
- `mlx.request.queue_seconds`
- `mlx.request.prefill_seconds`
- `mlx.request.prompt_tokens`
- `mlx.request.cached_prompt_tokens`
- `mlx.prompt_cache.hits.total`
- `mlx.prompt_cache.lookup_seconds`
- Signposts: `request.render`, `tokenize`, `admission`, `prompt_cache.lookup`,
  `prompt_cache.restore`, `memory_guard.check`, `prefill`, `first_token`

Trace template:

- Start with `Time Profiler`.
- Use `Metal System Trace` if prefill is GPU-heavy or there are command-buffer
  gaps.
- Use `System Trace` if wall-clock time is high but CPU samples are low.

Likely fixes:

- Shorten rendered prompt or tool schemas.
- Improve prompt-cache hit rate.
- Increase or adapt prefill chunk size if memory allows.
- Reduce queue contention or serialize only when memory pressure requires it.
- Check tokenization and template rendering hot paths.

### Low Generation Tokens Per Second

Signals to inspect:

- `mlx.request.decode_seconds`
- `mlx.request.inter_token_seconds`
- `mlx.request.generated_tokens`
- `mlx.tokens.generated.total`
- Speculative/MTP acceptance counters, when enabled.
- Grammar mask duration and rejection counters.
- Memory pressure and KV cache gauges.

Trace template:

- `Time Profiler` for sampler, grammar, token processing, and stream translation.
- `Metal System Trace` for GPU starvation, command-buffer gaps, or CPU/GPU
  synchronization.
- `Logging` when signpost intervals need closer inspection.

Likely fixes:

- Disable or optimize expensive grammar constraints.
- Check sampler configuration and token processors.
- Inspect dynamic KV quantization cost.
- Compare direct decode with speculative/MTP paths.
- Confirm streaming sinks are not back-pressuring the decode loop.

### Good TPS but Bad End-to-End Latency

Signals to inspect:

- `mlx.request.latency_seconds`
- `mlx.request.queue_seconds`
- `mlx.request.time_to_first_token_seconds`
- `mlx.request.decode_seconds`
- Active and queued request gauges.
- Stream translation span duration.

Trace template:

- `System Trace` for waiting, locks, scheduling, QoS, and I/O.
- `Swift Concurrency` for async stream or actor/executor contention.
- `Time Profiler` for CPU-bound post-processing.

Likely fixes:

- Reduce queuing and model-pool contention.
- Check stream consumer backpressure.
- Split expensive finalization work away from token emission.
- Avoid cache save work on the critical response path when possible.

### Prompt Cache Miss or Hit-Rate Regression

Signals to inspect:

- `mlx.prompt_cache.queries.total`
- `mlx.prompt_cache.hits.total`
- `mlx.prompt_cache.evictions.total`
- `mlx.request.cached_prompt_tokens`
- `mlx.prompt_cache.lookup_seconds`
- `mlx.prompt_cache.restore_seconds`
- `mlx.prompt_cache.save_seconds`
- Prompt-cache observability windows.

Trace template:

- `Time Profiler` for lookup/serialization CPU cost.
- `File Activity` for persistent cache I/O.
- `Allocations` when hot payload cache growth is suspicious.

Likely fixes:

- Check prompt renderer cache fingerprint stability.
- Check prompt style, tool schema, and assistant-history rendering changes.
- Inspect block alignment and partial-prefix reuse.
- Tune hot cache size and persistent cache budget.
- Add diagnostics for the first divergent token or cache-layout mismatch.

### Memory Guard Rejection or OOM Risk

Signals to inspect:

- `mlx.memory.current_bytes`
- `mlx.memory.limit_bytes`
- `mlx.memory.estimated_peak_bytes`
- `mlx.memory_guard.rejections.total`
- Model-pool resident bytes.
- KV cache bytes/entries.
- Prompt-cache hot bytes and SSD bytes.
- Memory guard tier and limit source.

Trace template:

- `Allocations` for heap and anonymous VM.
- `System Trace` for VM pressure and blocking.
- `Metal System Trace` for Metal memory limiters and GPU scheduling.
- MetricKit diagnostics in host apps for real-world hangs, exits, and memory
  exceptions.

Likely fixes:

- Lower max generated tokens or context size.
- Reduce concurrent requests.
- Adjust prefill chunking.
- Evict or unload resident models earlier.
- Reduce hot cache budget.
- Prefer persistent cache over hot in-memory cache for large prompts.

### Slow Model Load

Signals to inspect:

- `mlx.model.load_seconds`
- Model-pool loading/resident gauges.
- Resident weight bytes.
- File Activity trace for safetensors reads.
- Logs for model switching, corrupt cache restore, and oversized restores.

Trace template:

- `File Activity` for disk I/O.
- `Time Profiler` for parsing, header scanning, and weight loading CPU.
- `Allocations` for peak memory during load.

Likely fixes:

- Avoid unnecessary unload/reload churn through model-pool residency policy.
- Improve safetensors header scanning and shard loading.
- Keep model weights on MachineData and symlink into the repo, as this project
  already does.
- Prewarm models only when memory budget allows.

### Tool Calling or Structured Output Failure

Signals to inspect:

- `mlx.tool_calls.total`
- `mlx.tool_parser_failures.total`
- `mlx.grammar_rejections.total`
- Tool parser spans.
- Grammar mask spans.
- Finish reason.
- Request prompt style and tool dialect, as non-content attributes.

Trace template:

- `Time Profiler` for parser and grammar CPU cost.
- `Logging` for structured, privacy-safe parser state transitions.
- `Foundation Models` when debugging provider-path tool loops on macOS 27.

Likely fixes:

- Check tool dialect selection.
- Check renderer/tool schema compatibility.
- Add parser diagnostics for envelope, balanced JSON, structural tag, or native
  dialect failure stage.
- Ensure malformed tool calls still produce usage metrics.

### Provider Path Regression on macOS 27

Signals to inspect:

- Same request summary fields as the direct MLX path.
- Foundation Models usage snapshots.
- Provider bridge event translation spans.
- Tool usage and response usage events.

Trace template:

- `Foundation Models` first.
- `Logging` for provider bridge state.
- `Time Profiler` if bridge code appears CPU-bound.

Likely fixes:

- Compare direct MLX and provider-path metrics for the same real-model prompt.
- Inspect Foundation Models transcript rendering.
- Validate usage translation and tool event ordering.
- Check macOS 27-only API availability paths.

## Observability Report Bundle

For real-model tests and profiling runs, the ideal artifact is a small report
bundle next to the `.trace` file:

- `run.json`: model ID, runtime kind, profile, commit, OS version, Xcode version,
  host memory, model path fingerprint, test/example ID, and environment knobs.
- `request-summary.json`: final request metrics, cache counters, memory guard
  snapshots, finish reason, and error class.
- `events.jsonl`: optional structured trace events, excluding prompt/response
  content.
- `stdout`: generated text output or existing script stdout.
- `.trace`: Instruments trace, when captured.

This makes a regression review reproducible without opening Instruments first.
The JSON should answer: what ran, how many tokens, how long each phase took,
what cache did, what memory guard decided, and where to open the trace.

## Acceptance Criteria

Observability should be considered good enough only when these statements are
true:

- A one-request real-model run produces a request summary without content
  leakage.
- A prompt-cache real-model run shows miss on first run and hit/reused-token
  counters on second run.
- A memory guard rejection has a structured reason, limit source, estimated
  peak, current memory, and suggested subsystem.
- A generated Instruments trace can be aligned to request phases through
  signposts.
- A host app can export metrics without depending on internal test-only
  diagnostics.
- Metric names and labels are stable and low-cardinality.
- Optional detailed tracing can be disabled completely.
- Content capture, when added, is explicit, local, bounded, redacted/truncated,
  and off by default.

## Implementation Plan

### Phase 1: Stable In-Process Metrics

Add a small observability core:

- `MLXMetricsSnapshot`
- `MLXRequestMetrics`
- `MLXRuntimeMetrics`
- `MLXPromptCacheMetrics`
- `MLXMetricHistogram`
- `MLXObservabilitySink`

The sink should be optional and no-op by default. It should accept structured
events and final request summaries. Existing `ChunkMetrics`,
`MLXGenerationDiagnosticEvent`, `MLXPromptCacheObservabilitySnapshot`, and
`MLXModelPoolSnapshot` can feed this without changing generation behavior.

### Phase 2: OSLog and OSSignposter Bridge

Add an Apple-specific sink guarded by platform availability:

- Stable subsystem such as `org.mlxfoundationmodel`.
- Categories for generation, cache, memory, model pool, provider, tools,
  grammar, and streaming.
- Signpost intervals for major request phases.
- No prompt/response capture by default.
- Debug-only optional metadata for local traces.

This should be implemented as a thin adapter over the generic observability
sink, not woven directly through every runtime type.

### Phase 3: Repeatable Profiling Workflow

Extend the existing profiling workflow to:

- Record with `Foundation Models` for provider-path tests on macOS 27.
- Record with `Time Profiler`, `Metal System Trace`, `System Trace`,
  `Allocations`, `File Activity`, and `Logging` for direct MLX paths.
- Emit signpost names in the generated stdout or a small sidecar JSON so traces
  can be matched to model ID, example ID, generation token count, and cache mode.
- Keep traces in `.build/reports/profiles`.

### Phase 4: Host-App MetricKit Example

Add documentation or an example showing how a host app can:

- Create a macOS 27 `MetricManager`.
- Iterate `metricReports` and `diagnosticReports`.
- Persist or upload reports as JSON.
- Use state reporting to tag app states.
- Use package signposts so generation intervals appear in MetricKit custom
  signpost reports and Instruments traces.

Do not make MetricKit a core dependency of `MLXFoundationModel`.

### Phase 5: Optional Exporters

After the typed sink exists, add optional adapters:

- JSON snapshot exporter.
- Prometheus text formatter for server hosts.
- OpenTelemetry bridge for apps or services that already use OTel.

These should live as opt-in adapters, not in the runtime hot path.

## Testing Strategy

Use unit tests for the data model:

- Counters aggregate correctly.
- Histograms bucket values correctly.
- Snapshots do not expose prompt or response content.
- Prompt-cache diagnostic events map to cache metrics correctly.
- Memory guard diagnostics map to rejection counters and gauges correctly.
- Finish reasons map to request counters.

Use real-model tests sparingly:

- One smoke generation should assert final request metrics are populated.
- Prompt-cache real-model tests should assert cache hit/miss and reused-token
  counters.
- Provider-path tests on macOS 27 should verify that Foundation Models usage
  still produces final usage and that signpost instrumentation does not break
  streaming.

Use profiling manually or in scheduled workflows:

- Run `scripts/profile-real-model.sh` with short time limits.
- Keep one baseline trace per important profile only when investigating a
  regression.
- Do not commit traces.

## Near-Term Recommendation

The next concrete implementation should be:

1. Introduce a no-op-by-default `MLXObservabilitySink`.
2. Emit one final structured request summary from `MLXSession`.
3. Add signpost intervals for model load, request rendering, prompt-cache
   lookup/restore/save, prefill, decode, and stream translation.
4. Add tests proving metrics do not contain prompt or response content.
5. Update `scripts/profile-real-model.sh` docs to recommend template selection
   by symptom.

That gives us the Apple-native profiling benefits immediately while keeping the
package standalone and exporter-neutral.
