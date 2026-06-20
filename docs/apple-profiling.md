# Apple Profiling

Use release builds and short, repeatable traces. Keep traces under
`.build/reports/profiles` so they stay out of git and can be deleted when disk
space is tight.

Apple guidance relevant to this package:

- Foundation Models has its own Instruments template for profiling app
  interactions with the framework:
  <https://developer.apple.com/documentation/foundationmodels/analyzing-the-runtime-performance-of-your-foundation-models-app>
- Metal System Trace and GPU counters are the right view for GPU timelines,
  memory limiters, and command-buffer behavior:
  <https://developer.apple.com/videos/play/wwdc2020/10603/>
- Xcode's Metal performance guide is the starting point for deeper GPU
  investigations:
  <https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app>
- Memory profiling should be separated from token throughput profiling:
  <https://developer.apple.com/documentation/xcode/gathering-information-about-memory-use>
- Model loading is storage-heavy. Apple recommends asynchronous, granular
  resource loading to keep SSD and unified memory throughput busy:
  <https://developer.apple.com/videos/play/wwdc2022/10104/>

## Local Workflow

List available templates:

```sh
xcrun xctrace list templates
```

Profile the real playground with a small model:

```sh
scripts/profile-real-model.sh
```

Profile GPU work:

```sh
MLX_PROFILE_TEMPLATE='Metal System Trace' scripts/profile-real-model.sh
```

Profile model-load I/O:

```sh
MLX_PROFILE_TEMPLATE='File Activity' MLX_PROFILE_TIME_LIMIT=15s scripts/profile-real-model.sh
```

Profile memory:

```sh
MLX_PROFILE_TEMPLATE='Allocations' scripts/profile-real-model.sh
```

The script builds `FoundationModelsPlayground` in release mode, runs one real
example, captures generated token output, records a `.trace`, and exports a
trace table of contents when `xctrace export` supports it.

Use the Foundation Models template when running on a macOS 27 host through the
Apple provider API. Use Time Profiler, Metal System Trace, File Activity, and
Allocations for the direct MLX path because those traces expose Swift runtime
cost, Metal/MLX GPU scheduling, model-load I/O, and memory pressure directly.
