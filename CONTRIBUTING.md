# Contributing

Keep the package standalone: no app-specific dependencies, no model weights in
git, and real-model tests opt-in.

Before opening changes:

```sh
make test
```

When touching model loading, generation, prompt rendering, constraints, caching,
or tool calls, also run at least:

```sh
make test-demo-model
```

For representative architecture coverage:

```sh
make download-main-models
make test-main-architectures
```

Use Swift Testing for new tests. Put downloaded models in ignored `.models/` or
set `MLX_TEST_MODELS_DIR`.
