# Model Parity

`MLXFoundationModel` should support the same text-generation model families as
`mlx-lm`, exposed through Apple's Foundation Models shape.

The package is not trying to become a general MLX server. oMLX, Ollama, and LM
Studio cover server lifecycle, dashboards, routing, embeddings, reranking, OCR,
and VLM serving. This package should stay focused on app-facing Swift APIs:

- Foundation Models-style sessions.
- Streaming text.
- Tool calling.
- Structured output.
- Prompt caching and model residency.
- Metrics, logs, and Instruments signposts.

## Target

The parity target is `mlx-lm` text-generation architecture support. New model
families should land with:

- A project-owned Swift implementation.
- Focused architecture tests.
- A real-model catalog entry when a runnable checkpoint exists.
- Serialized E2E coverage that keeps one model loaded at a time.

Non-text families are tracked separately. They should not be added unless there
is a clear Apple interface shape for them in this package.

## Current Gaps

Snapshot: `ml-explore/mlx-lm` model modules on 2026-07-01.

Highest priority:

- `kimi_linear`
- `step3p5`

MoE and recent family gaps:

- `bailing_moe_linear`
- `afm7`
- `nemotron-nas`

Legacy and compatibility gaps:

- `recurrent_gemma`

## Out Of Scope For This Track

These are useful in the MLX ecosystem, but they are not part of the
Foundation Models-compatible text target:

- VLM: `qwen2_vl`, `qwen3_vl`, `qwen3_vl_moe`, `pixtral`, `kimi_vl`, `lfm2-vl`.
- OCR: `dots1`, DeepSeek-OCR, PaddleOCR.
- Embeddings and rerankers: BERT, BGE, ModernBERT, XLM-RoBERTa.

They can be revisited if this package grows a matching Apple-facing API surface.

## Guardrail

`MLXLMParityTests` keeps the text parity gap explicit. When a model family is
implemented, remove it from the deferred list, add focused tests, and add a
real-model E2E entry when there is a checkpoint that fits the test host.
