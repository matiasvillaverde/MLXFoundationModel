# Provenance and Benchmarks

Last updated: 2026-07-01

## Provenance

This repository is MIT licensed. No audited file carries a source-port note.

Audit of `Sources/MLXLocalModels/Common` and `Sources/MLXLocalModels/MLXLLM`:

| Area | Files | Notes |
| --- | ---: | --- |
| Apple source-level notices | 0 | No Apple source notices remain in the audited paths. |
| Explicit source-port markers | 0 | Counted from real provenance markers, not ordinary comments that say "based on". |
| Files with no source-port marker | 123 | Safe area for normal refactors. |

Replaced in the current independence pass:

| File | Replacement scope |
| --- | --- |
| `Sources/MLXLocalModels/Common/StringOrNumber.swift` | Config scalar decoding and token-id field helpers. |
| `Sources/MLXLocalModels/Common/BaseConfiguration.swift` | Base config and mixed per-layer quantization decoding. |
| `Sources/MLXLocalModels/Common/MLXConfigFileDecoder.swift` | Shared JSON5 config-file loading into dictionary form for profile and memory estimation paths. |
| `Sources/MLXLocalModels/Common/ModelConfiguration.swift` | Model identity, tokenizer overrides, local directory resolution, and generation token defaults. |
| `Sources/MLXLocalModels/Common/ModelContainer.swift` | Actor-owned model context and prompt-cache access. |
| `Sources/MLXLocalModels/Common/ModelFactory.swift` | Model context tokenization, factory dispatch, fallback errors, and trampoline registry. |
| `Sources/MLXLocalModels/Common/AbstractModelRegistry.swift` | Thread-safe model configuration lookup and fallback creation. |
| `Sources/MLXLocalModels/Common/ModelTypeRegistry.swift` | Thread-safe model type constructor lookup and unsupported-type reporting. |
| `Sources/MLXLocalModels/Common/Evaluate.swift` | Generation parameter normalization, sampler planning, processor planning, token iteration control, and generation result timing. |
| `Sources/MLXLocalModels/Common/GenerationConstants.swift` | Shared generation, cache, and sampling defaults. |
| `Sources/MLXLocalModels/Common/KVCache.swift` | Cache protocols, dense/rotating/chunked/quantized KV storage, prompt-cache serialization, quantized attention, and runtime cache quantization. |
| `Sources/MLXLocalModels/Common/LanguageModel.swift` | Core model input/output contracts, default forwarding, greedy helpers, and cache creation. |
| `Sources/MLXLocalModels/Common/Load.swift` | Model artifact matching, deterministic safetensor discovery, and weight loading. |
| `Sources/MLXLocalModels/Common/LoRA+Layers.swift` | Dense and quantized LoRA replacement layers, adapter initialization, freeze policy, and fusion. |
| `Sources/MLXLocalModels/Common/Module+Extensions.swift` | Logical parameter counting for dense and quantized modules. |
| `Sources/MLXLocalModels/Common/RoPEApplication.swift` | Scalar and batch RoPE offset selection for KV caches. |
| `Sources/MLXLocalModels/Common/SuScaledRoPE.swift` | LongRoPE factor planning, short/long frequency selection, scalar and batch offsets, and non-rotary tail preservation. |
| `Sources/MLXLocalModels/Common/Tokenizer.swift` | Tokenizer loading, tokenizer-class rewriting, replacement registry, and streaming detokenization. |
| `Sources/MLXLocalModels/Common/Phi3SmallTiktokenTokenizer.swift` | Phi-3-small tiktoken vocabulary loading, byte-pair encoding, special-token handling, and chat-template rendering. |
| `Sources/MLXLocalModels/Common/QwenTiktokenTokenizer.swift` | Qwen tiktoken vocabulary loading, byte-level BPE, special-token handling, and default chat rendering. |
| `Sources/MLXLocalModels/Common/SentencePieceModelTokenizer.swift` | SentencePiece model-file parsing, duplicate-piece tolerant BPE lookup, byte fallback, special-token splitting, and InternLM-style chat rendering. |
| `Sources/MLXLocalModels/Common/RWKV7Tokenizer.swift` | RWKV7 longest-match byte tokenizer loading, special-token handling, grammar-vocabulary fallback, and UTF-8 decode recovery. |
| `Sources/MLXLocalModels/MLXLLM/LLMModel.swift` | Default text-model prefill chunking and adaptive prefill integration. |
| `Sources/MLXLocalModels/MLXLLM/LLMModelFactory.swift` | LLM type registration, alias grouping, model load progress, generation-token resolution, and trampoline factory. |
| `Sources/MLXLocalModels/MLXLLM/Cohere2.swift` | Cohere2 grouped attention, hybrid sliding/full attention schedule, mixed cache planning, tied output head, greedy-token fast path, stale rotary cleanup, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/RWKV7.swift` | RWKV7 time-mixing, channel-mixing, recurrent WKV state updates, Metal recurrence dispatch, tied-head fallback, greedy-token fast path, and cache planning. |
| `Sources/MLXLocalModels/MLXLLM/GPT2.swift` | GPT-2 learned position embeddings, cache-aware position IDs, pre-norm attention and MLP blocks, raw Transformers sanitizer, tied output head, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GPTBigCode.swift` | GPT-BigCode learned position embeddings, multi-query packed attention, raw Transformers sanitizer, tied/untied output heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GPTNeoX.swift` | GPT-NeoX partial-RoPE packed attention, parallel and sequential residual blocks, raw Transformers sanitizer, tied/untied output heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/StableLM.swift` | StableLM partial-RoPE attention, optional per-head q/k LayerNorm, sequential and parallel residual blocks, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GLM.swift` | GLM grouped-query attention, traditional RoPE, fused SwiGLU feed-forward blocks, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/HunyuanV1Dense.swift` | Hunyuan V1 Dense grouped-query attention, dynamic-alpha RoPE, q/k RMSNorm, tied-head cleanup, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Helium.swift` | Helium grouped attention, traditional RoPE planning, SwiGLU feed-forward blocks, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/BailingMoe.swift` | Bailing MoE attention layout, sparse routing plan, grouped expert selection, expert packing, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Mistral3Text.swift` | Mistral 3 attention layout, Llama 4 position scaling, full/sliding layer scheduling, cache planning, VLM weight unwrapping, greedy-token fast path, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/MiniCPM.swift` | MiniCPM attention layout, residual/embedding/logit scaling plans, stable checkpoint keys, tied-head sanitizing, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/MiniCPM3.swift` | MiniCPM3 MLA attention layout, LongRoPE runtime frequencies, residual/embedding/logit scaling, tied-head sanitizing, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/MiniMax.swift` | MiniMax attention layout, sparse routing plan, expert weight packing, stable checkpoint keys, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/MiMoV2Flash.swift` | MiMo v2 Flash full/sliding attention layout, layer scheduling, grouped routing, attention sinks, expert packing, per-layer cache and KV-head planning, greedy-token fast path, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Olmo.swift` | OLMo LayerNorm-without-affine blocks, GQA-compatible RoPE attention, tied/untied heads, legacy packed-key sanitizing, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/OlmoE.swift` | OLMoE attention layout, sparse routing plan, expert packing, stable checkpoint keys, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GatedDelta.swift` | Gated-delta layout planning, decay calculation, Metal kernel dispatch, ops fallback, mask handling, unsupported-shape fallback, and deterministic inactive-token output. |
| `Sources/MLXLocalModels/MLXLLM/Gemma.swift` | Gemma RMSNorm, residual clipping, attention layout, decoder block, backbone, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Gemma2.swift` | Gemma2 soft-capped attention, grouped KV expansion, decoder block, backbone, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Internlm2.swift` | InternLM2 packed attention, dynamic RoPE planning, decoder blocks, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/InternLM3.swift` | InternLM3 grouped-query attention, dynamic RoPE planning, checkpoint-compatible module keys, tied-head cleanup, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/DeepseekV3.swift` | DeepSeek V3 attention layout, YaRN planning, grouped MoE routing, checkpoint key normalization, cache dimensions, greedy-token fast path, sanitizer packing, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Deepseek.swift` | DeepSeek MoE attention layout, top-k expert routing, shared experts, packed expert checkpoint remapping, tied-head cleanup, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Mixtral.swift` | Mixtral grouped-query attention, top-k sparse routing, expert tensor packing, tied-head cleanup, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GLM4MOE.swift` | GLM4 MoE attention layout, layer plan, grouped sparse routing, expert packing, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/AfMoE.swift` | AfMoE full/sliding attention layout, layer schedule, grouped sparse routing, expert packing, mixed cache planning, tied/untied heads, greedy-token fast path, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Qwen3Next.swift` | Qwen3Next layer scheduling, gated full attention, gated-delta linear attention, MoE routing, expert packing, mixed cache planning, tied/untied heads, greedy-token fast path, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GraniteMoeHybrid.swift` | Granite MoE Hybrid typed layer scheduling, attention and Mamba layout planning, MoE routing, shared/dense feed-forward remapping, mixed cache planning, tied/untied heads, greedy-token fast path, SSM mask use, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Mellum.swift` | Mellum full/sliding attention scheduling, per-layer RoPE selection, q/k RMSNorm, sparse MoE routing, expert packing, mixed cache planning, tied/untied heads, greedy-token fast path, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/SeedOSS.swift` | Seed OSS grouped-query attention, default RoPE scaling, SwiGLU feed-forward blocks, tied/untied heads, greedy-token fast path, cache dimensions, sanitizer cleanup, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Nemotron.swift` | Nemotron LayerNorm1P, partial-RoPE grouped-query attention, squared-ReLU feed-forward blocks, tied/untied heads, greedy-token fast path, cache dimensions, sanitizer cleanup, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/NemotronH.swift` | Nemotron H typed block scheduling, attention and Mamba layout planning, squared-ReLU dense and routed feed-forward paths, grouped expert routing, mixed cache planning, tied/untied heads, greedy-token fast path, SSM mask use, sanitizer packing, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GLM4MoELite.swift` | GLM4 MoE Lite and GLM DSA attention planning, grouped routing, DSA cache layout, multi-head projection quantization, tied/untied heads, greedy-token fast path, sanitizer packing, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Llama.swift` | Llama/Mistral attention layout, linear/dynamic/Llama 3 RoPE planning, decoder block, backbone, tied/untied output heads, greedy-token fast path, config validation, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Exaone.swift` | EXAONE 3.x grouped-query attention, llama3 RoPE scaling, SwiGLU feed-forward blocks, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Phi3.swift` | Phi3 packed QKV attention, RoPE/LongRoPE planning, decoder block, backbone, tied/untied output heads, greedy-token fast path, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Phi3Small.swift` | Phi-3-small packed QKV attention, block-sparse attention planning, MuP scaling, GeGELU feed-forward blocks, tied output head, dummy-token suppression, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Phi.swift` | Phi attention layout, decoder block, backbone, greedy-token fast path, configuration defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/PhiMoE.swift` | Phi MoE attention layout, LongRoPE planning, router planning, guarded expert packing, stable checkpoint keys, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/SwitchLayers.swift` | Expert dispatch permutation, SwitchGLU routing, dense/quantized expert projection, and sorted-dispatch restoration. |
| `Sources/MLXLocalModels/MLXLLM/Granite.swift` | Granite attention layout, RoPE scaling plan, residual/embedding/logit scaling, tied/untied heads, greedy-token fast path, config defaults, stable checkpoint keys, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/GraniteMoE.swift` | GraniteMoE attention layout, RoPE scaling plan, top-k expert routing, packed checkpoint remapping, tied/untied heads, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Ernie4_5.swift` | ERNIE 4.5 attention layout, explicit head-dimension override support, tied/untied heads, greedy-token fast path, config defaults, stable checkpoint keys, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Olmo2.swift` | OLMo2 attention layout, q/k normalization, checkpoint-compatible `model.*` keys, tied-head sanitizing, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Olmo3.swift` | OLMo3 sliding/full layer schedule, attention layout, q/k norm, YaRN-vs-sliding RoPE selection, cache layout, tied/untied heads, greedy-token fast path, sanitizer, config defaults, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Qwen35.swift` | Qwen3.5 text config decoding, explicit layer schedule, attention and linear-attention layouts, cache planning, native MTP gating, tied/untied heads, greedy-token fast path, sanitizer, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Qwen35MoE.swift` | Qwen3.5 MoE top-level config fallback, shared top-level weight mapping, expert projection remapping, and sanitizer delegation. |
| `Sources/MLXLocalModels/MLXLLM/Qwen2MoE.swift` | Qwen2 MoE attention layout, sparse routing, shared expert path, expert packing, tied-head sanitizing, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Qwen.swift` | Qwen packed QKV attention, byte-level tokenizer support, SwiGLU feed-forward blocks, tied-head cleanup, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/LFM2MoE.swift` | LFM2 MoE typed layer planning, attention/convolution layouts, router planning, guarded decoder dispatch, cache/KV-head planning, sanitizer packing, greedy path preservation, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Mamba.swift` | Mamba selective state-space decoding, depthwise convolution cache updates, tied/untied heads, checkpoint weight cleanup, greedy-token fast path, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Mamba2.swift` | Mamba2 state-space mixer, gated RMS norm, derived intermediate dimensions, cache updates, tied-head cleanup, greedy-token fast path, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/NanoChat.swift` | NanoChat attention layout, custom rotary-frequency plan, RMSNorm/softcap planning, stable transformer checkpoint keys, greedy-token fast path, cache dimensions, and LoRA target discovery. |
| `Sources/MLXLocalModels/MLXLLM/Lora+Data.swift` | LoRA JSONL/text data lookup and parsing. |
| `Sources/MLXLocalModels/MLXLLM/LoraTrain.swift` | LoRA batching, conversion/fusion, masked loss, evaluation, save/load, and training progress. |

## Recent Code Changes

Current independence pass:

- Replaced the shared model configuration and registry files listed above.
- Removed an unused prompt-preparation closure from `ModelConfiguration`; its only registry call site never affected runtime behavior because the closure was not stored or used.
- Added focused SwiftTesting coverage for remote and local model identity, equality, fallback configuration creation, replacement registration, constructor lookup, unsupported model errors, and concurrent registry writes.
- Replaced default text-model prefill support with focused coverage for constants, default chunk size, explicit window chunking, and prompt-tail preservation.
- Replaced module parameter counting with a single-pass implementation and focused coverage for dense, embedding, and quantized module leaves.
- Replaced RoPE offset selection with focused coverage for nil-cache, scalar-cache, and per-row batch-cache paths.
- Replaced Su-scaled RoPE with explicit LongRoPE planning, short/long factor validation, context-length frequency selection, and focused plan/call coverage.
- Replaced SwitchLayers with explicit expert routing permutations, shared dense/quantized bias handling, deterministic sorted-dispatch coverage, and real MoE validation.
- Replaced model container ownership with focused coverage for context updates, perform forwarding, legacy overload compatibility, and prompt-cache mutation.
- Replaced model factory dispatch with focused coverage for chat-template tokenization, rendered/cache prompt encoding, factory fallback, final-error propagation, and missing-factory errors.
- Replaced tokenizer support with focused coverage for tokenizer-class rewriting, registry updates, streaming deltas, newline resets, and incomplete Unicode boundaries.
- Added SentencePiece model-file tokenizer fallback with duplicate-piece tolerant lookup, bounded special-token splitting, grammar-vocabulary fallback, and focused synthetic fixture coverage.
- Added RWKV7 longest-match byte tokenizer support with grammar-vocabulary fallback and focused synthetic fixture coverage.
- Replaced LoRA data loading with focused coverage for lookup precedence, JSONL parsing, text lines, missing files, and unsupported file types.
- Replaced LoRA layer adapters with focused coverage for dense/quantized conversion, adapter-only training, no-op initialization, fusion, and quantized mode preservation.
- Replaced LoRA training helpers with focused coverage for shifted causal batches, prediction-length masking, weighted evaluation, adapter conversion/fusion, and quantized dequantize fusion.
- Replaced LLM model factory registration with grouped aliases, testable generation-token resolution, and focused coverage for alias registration plus EOS/suppress-token precedence.
- Added Cohere2 with hybrid sliding/full attention scheduling, LayerNorm bias handling, mixed cache planning, tied output head, greedy-token fast path, focused coverage, and real Tiny Aya Global validation.
- Replaced generation parameter and logit plan assembly with normalized inputs, explicit sampler/processor planning, and focused coverage for sampler selection plus active processor construction.
- Replaced LanguageModel core contracts with focused coverage for input slicing, media wrappers, default forwarding, greedy helpers, sanitize fallback, and KV-cache creation.
- Replaced model loading support with deterministic safetensor discovery, explicit directory errors, and focused coverage for recursive discovery, case-insensitive extensions, and missing directories.
- Added shared JSON5 config loading for profile and memory-guard paths, covering checkpoint configs that use values such as bare `Infinity`.
- Replaced KV cache internals with shared append planning, explicit dense and quantized state wrappers, active-window chunk trimming, stable prompt-cache layout serialization, and focused cache growth/chunk/quantization coverage.
- Replaced Phi with an explicit attention layout, project-owned module structure, config defaults, greedy-token fast path, and focused layout/config/LoRA coverage.
- Replaced InternLM2 with packed-attention layout, type-specific RoPE scaling, greedy-token fast path, packed LoRA targeting, and focused layout/RoPE/config coverage.
- Added InternLM3 with grouped-query attention, dynamic RoPE scaling, checkpoint-compatible keys, greedy-token fast path, SentencePiece tokenizer coverage, and targeted real-model validation.
- Added Hunyuan V1 Dense with dynamic-alpha RoPE, q/k RMSNorm, tied-head cleanup, greedy-token fast path, focused coverage, and real Hunyuan MT validation.
- Replaced Gemma with a shared project-owned norm, explicit attention layout, stable checkpoint keys, greedy-token fast path, and focused config/layout/LoRA coverage.
- Replaced Gemma2 with soft-capped attention layout, grouped KV expansion, greedy-token fast path, stable checkpoint keys, and focused config/layout/LoRA coverage.
- Replaced Phi3 with packed QKV layout, explicit RoPE/LongRoPE planning, tied/untied output handling, greedy-token fast path, and focused config/layout/LoRA coverage.
- Added Phi-3-small with packed QKV attention, block-sparse attention planning, MuP scaling, tiktoken tokenizer support, grammar-vocabulary fallback, dummy-token suppression, greedy-token fast path, and focused config/layout/forward/tokenizer coverage.
- Replaced Llama with explicit Llama/Mistral layout, project-owned RoPE planning for linear, dynamic, and Llama 3 scaling, tied/untied output handling, greedy-token fast path, and focused config/layout/LoRA coverage.
- Replaced DeepSeek V3 with explicit attention, YaRN, and MoE routing plans; fixed empty KV-cache dimensions; corrected adapter targets; packed expert weights in the sanitizer; and added focused config/layout/routing/forward/sanitizer coverage.
- Added DeepSeek MoE with explicit GQA attention, RoPE planning, dense-vs-sparse layer scheduling, shared experts, top-k routing, packed expert checkpoint remapping, tied-head cleanup, greedy-token fast path, and focused config/layout/routing/forward/sanitizer coverage.
- Added Mixtral with explicit GQA attention, top-k expert routing, packed expert checkpoint remapping, tied-head cleanup, greedy-token fast path, and Mistral-family prompt-style inference for Mixtral checkpoints.
- Replaced Granite with an explicit attention layout, linear RoPE scaling plan, stable checkpoint-compatible `model.*` parameter keys, tied/untied output handling, greedy-token fast path, and focused config/layout/forward/LoRA coverage.
- Replaced ERNIE 4.5 with an explicit attention layout, head-dimension fallback and override handling, stable checkpoint-compatible `model.*` parameter keys, tied/untied output handling, greedy-token fast path, and focused config/layout/forward/LoRA coverage.
- Replaced OLMo3 with explicit sliding/full attention scheduling, q/k normalization, YaRN-vs-sliding RoPE selection, cache layout, tied/untied output handling, greedy-token fast path, and focused config/layout/cache/LoRA coverage.
- Replaced Qwen3.5 text and MoE wrappers with explicit config schedule decoding, validated layout plans, shared top-level weight mapping, native MTP gating, greedy-token fast path, and focused schedule/layout/cache/MTP coverage.
- Added Qwen2 MoE with explicit Qwen2 attention, softmax top-k routing, shared expert gating, expert tensor packing, tied-head cleanup, greedy-token fast path, and focused config/layout/forward/sanitizer coverage.
- Added Qwen with packed QKV attention, byte-level tiktoken fallback, SwiGLU feed-forward blocks, greedy-token fast path, grammar-vocabulary fallback, and focused config/layout/cache/forward/tokenizer coverage.
- Replaced LFM2 MoE with typed layer planning, explicit attention/convolution layouts, guarded layer dispatch, complete expert packing, attention-only LoRA targeting, and focused plan/cache/sanitizer coverage.
- Replaced Phi MoE with explicit attention and router plans, centralized LongRoPE support, safe expert packing, stable `model.*` parameter keys, greedy-token fast path, and focused config/layout/forward/sanitizer coverage.
- Replaced MiniCPM with explicit attention and scaling plans, registered checkpoint-compatible module keys, tied-head sanitizing, greedy-token fast path, and focused config/layout/forward/sanitizer coverage.
- Added MiniCPM3 with explicit MLA attention, non-loadable LongRoPE runtime frequencies, MiniCPM scaling, tied-head cleanup, greedy-token fast path, and focused config/layout/forward/sanitizer coverage.
- Replaced Mistral 3 text with explicit attention and layer-schedule plans, Llama 4 position scaling, VLM weight unwrapping, mixed cache creation, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Replaced OLMo2 with explicit attention layout, q/k normalization, stable `model.*` parameter keys, tied-head sanitizing, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added EXAONE 3.x with explicit grouped-query attention, llama3 RoPE scaling, checkpoint-compatible nested attention keys, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Replaced NanoChat with explicit attention, rotary-frequency, RMSNorm, and logit-softcap plans; preserved transformer checkpoint keys; added greedy-token fast path, simple cache creation, and focused config/layout/forward/softcap coverage.
- Replaced GatedDelta with explicit shape planning, deterministic mask semantics, safe unsupported-shape fallback, project-owned Metal recurrence dispatch, and focused decay/layout/fallback coverage.
- Replaced MiniMax with explicit attention and sparse-routing plans, stable `model.*` checkpoint keys, expert packing, tied-head handling, greedy-token fast path, and focused config/layout/forward/sanitizer coverage.
- Replaced OLMoE with explicit attention and sparse-routing plans, stable `model.*` checkpoint keys, expert packing, tied-head handling, greedy-token fast path, and focused config/layout/routing/forward/sanitizer coverage.
- Replaced Bailing MoE with explicit attention, layer, routing, and expert-packing plans; fixed grouped routing edge cases; added tied-head handling, greedy-token fast path, cache creation, LoRA target discovery, and focused config/routing/forward/sanitizer coverage.
- Replaced MiMo v2 Flash with explicit full/sliding attention and layer-schedule plans, safer grouped routing, attention-sink handling, expert packing, per-layer cache/KV-head planning, greedy-token fast path, and focused config/layout/routing/cache/forward/sanitizer coverage.
- Replaced GLM4 MoE with explicit attention, layer, and grouped-routing plans, safer correction-bias routing, expert packing, tied-head cleanup, greedy-token fast path, and focused config/layout/routing/cache/forward/sanitizer coverage.
- Replaced AfMoE with explicit full/sliding attention, layer, routing, and expert-packing plans; fixed grouped routing edge cases; added mixed cache creation, tied-head cleanup, greedy-token fast path, LoRA target discovery, and focused config/layout/routing/cache/forward/sanitizer coverage.
- Replaced Qwen3Next with explicit layer, attention, linear-attention, MoE, sanitizer, and expert-packing plans; fixed later-layer expert packing; added robust mixed-cache mask selection, tied-head cleanup, greedy-token fast path, LoRA target discovery, and focused config/layout/cache/forward/sanitizer coverage.
- Replaced Granite MoE Hybrid with typed layer, attention, Mamba, MoE, sanitizer, and cache plans; removed the local no-op SSM mask; fixed later-layer MoE/shared-MLP sanitizer remapping; added tied-head cleanup, greedy-token fast path, LoRA target discovery, and focused config/layout/cache/forward/sanitizer coverage.
- Added Mellum with full/sliding attention scheduling, YaRN/default per-layer RoPE selection, q/k RMSNorm, sparse MoE routing, sidecar-aware expert packing, mixed cache planning, greedy-token fast path, LoRA target discovery, and focused config/layout/cache/forward/sanitizer coverage.
- Added Nemotron with checkpoint-compatible LayerNorm1P, partial-RoPE grouped attention, squared-ReLU MLP blocks, tied-head cleanup, greedy-token fast path, LoRA target discovery, and focused config/layout/norm/cache/forward/sanitizer coverage.
- Replaced Nemotron H with typed block, cache, attention, Mamba, MoE, and sanitizer plans; fixed array-form `time_step_limit` decoding; added grouped routing, tied-head cleanup, greedy-token fast path, LoRA target discovery, and focused config/layout/routing/cache/forward/sanitizer coverage.
- Replaced GLM4 MoE Lite with explicit attention, DSA, layer, routing, projection, and sanitizer plans; registered layers through `@ModuleInfo`; added tied-head cleanup, greedy-token fast path, LoRA target discovery, and focused schedule/layout/routing/cache/sanitizer coverage.
- Added Seed OSS with grouped-query attention, default RoPE handling, SwiGLU feed-forward blocks, tied-head cleanup, greedy-token fast path, LoRA target discovery, focused config/layout/cache/forward/sanitizer coverage, and a fit-on-32GB 2-bit real-model catalog entry.
- Added GPT-2 with learned position embeddings, cache-aware position IDs, raw Transformers and MLX-prefixed sanitizer paths, tied output head, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added GPT-BigCode with learned position embeddings, multi-query packed attention, raw Transformers key mapping, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added GPT-NeoX with partial-RoPE packed attention, raw Transformers Pythia key mapping, parallel and sequential residual paths, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added StableLM with partial-RoPE attention, optional per-head q/k LayerNorm matching upstream checkpoint keys, sequential and parallel residual paths, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added GLM with grouped-query attention, traditional RoPE, fused SwiGLU feed-forward blocks, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added OLMo with affine-free LayerNorm, RoPE attention, HF and legacy mlx-lm config decoding, packed QKV/SwiGLU weight splitting, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added Mamba with selective state-space recurrence, depthwise convolution cache updates, alias-compatible config decoding, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added Mamba2 with gated state-space recurrence, derived intermediate sizing for compact configs, JSON5 checkpoint profile loading, CPU-safe SSM fallback, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added Helium with grouped attention, traditional RoPE, SwiGLU feed-forward blocks, tied-head cleanup, greedy-token fast path, and focused config/layout/cache/forward/sanitizer coverage.
- Added RWKV7 with project-owned time-mixing, channel-mixing, recurrent WKV cache updates, a Metal recurrence path, greedy-token fast path, focused architecture/tokenizer coverage, and real Goose 0.1B validation.

Previous performance pass:

`Sources/MLXLocalModels/Common/AttentionUtils.swift` now separates the normal
attention path from the shared-KV path. When a caller only needs attention
output, quantized KV caches no longer materialize dequantized key and value
arrays that are immediately discarded.

This should help when runtime KV quantization is enabled. The real-model sweep
below is mainly a regression check for normal model execution.

## Environment

| Field | Value |
| --- | --- |
| Mac | MacBook Pro, Mac14,5 |
| Chip | Apple M2 Max |
| CPU cores | 12: 8 performance, 4 efficiency |
| Unified memory | 32 GB |
| macOS | 27.0, build 26A5353q |
| Xcode | 27.0, build 27A5194q |
| Swift | 6.4 |
| Model storage | `.build/test-models` |

Command:

```sh
MLX_TEST_MODELS_DIR="$PWD/.build/test-models" \
MLX_HOST_MEMORY_GB=32 \
MLX_REAL_MODEL_GENERATION_TOKENS=8 \
MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS=240 \
MLX_REAL_MODEL_TIMEOUT_SECONDS=1200 \
MLX_REAL_MODEL_FEATURE_TIMEOUT_SECONDS=900 \
CONFIGURATION=release \
make test-all-architectures
```

## E2E Result

The current all-architecture sweep passed for every model selected by the memory
gate. The test runner selected 78 local models and skipped 9 oversized models on
this 32 GB host. Each selected model ran serialized generation, rendered session
requests, and token-level grammar constraint checks.

A follow-up serialized `main` sweep on 2026-07-01 selected 42 downloadable
models, including Cohere2, DeepSeek, DeepSeek V2, EXAONE 3.5, GraniteMoE,
Helium, Hunyuan V1 Dense, InternLM3, Jamba, Mamba, Mamba2, Mixtral, Phixtral,
Qwen, RWKV7, Seed OSS, and GLM, and passed generation, rendered session, token
grammar, and configured stress checks.

The current sweep adds 32 GB-friendly checkpoints for `qwen3_moe`, `mistral`,
`gpt_oss`, `qwen3_5_moe`, and `nemotron_h`. These entries also run the stress
test, which preloads one session and repeats generation on that same session.

`glm4_moe`, `solar_open`, `glm4_moe_lite`, and pure `nemotron` are still
registry-only in the catalog, and no exact `glm4-moe`, `solar-open`,
`glm4-moe-lite`, or `nemotron` checkpoint directory is present locally. Solar
Open uses the same GLM4 MoE implementation path that mlx-lm exposes for
`solar_open`. The smallest published MLX Solar Open checkpoint found during
this pass was `mlx-community/Solar-Open-100B-4bit`; `hf download --dry-run`
reported twelve weight shards of about 57 GB total, so it is intentionally left
out of the 32 GB E2E sweep. Current small Nemotron MLX search hits are
`nemotron_h` or `llama` configs, so pure `nemotron` stays registry-only until a
fit checkpoint is available.

Qwen2 MoE parity was added with
`mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit`. The checkpoint is 7.9 GB on disk
and passed targeted real-model generation, rendered session requests, token
grammar constraints, and the serialized `main` architecture sweep on this host.

Qwen parity was added with `Qwen/Qwen-1_8B`. The checkpoint is 3.4 GB on disk
and passed targeted release generation, rendered session requests, token grammar
constraints, stress generation, and the serialized `main` architecture sweep on
this host.

GLM parity was added with `zai-org/glm-edge-1.5b-chat`. The checkpoint is 3.0 GB
on disk and passed targeted release generation, rendered session requests, token
grammar constraints, stress generation, and the serialized `main` architecture
sweep on this host.

Hunyuan V1 Dense parity was added with `tencent/Hunyuan-MT-7B`. The checkpoint
is 15 GB on disk and passed targeted release generation, rendered session
requests, token grammar constraints, stress generation, and the serialized
`main` architecture sweep on this host.

Phixtral parity was added with `mlabonne/phixtral-2x2_8`. The checkpoint is
8.3 GB on disk and passed targeted release generation, rendered session
requests, token grammar constraints, stress generation, and the serialized
`main` architecture sweep on this host.

Cohere2 parity was added with `Siarhei/tiny-aya-global-4bit`. The checkpoint is
1.8 GB on disk and passed targeted release generation, rendered session
requests, token grammar constraints, stress generation, and the serialized
`main` architecture sweep on this host.

RWKV7 parity was added with `chinoll/rwkv7-g1d-0.1b`. The checkpoint is
375 MB on disk and passed targeted release generation, rendered session
requests, token grammar constraints, stress generation, the serialized `main`
architecture sweep, and the release `all` architecture sweep on this host.

MiniCPM3 parity was added with `mlx-community/MiniCPM3-4B-4bit`. The checkpoint
is 2.2 GB on disk and passed targeted real-model generation, rendered session
requests, and token grammar constraints on this host.

GPT-2 parity was added with `mlx-community/gpt2-base-mlx`. The checkpoint is
475 MB on disk and passed targeted real-model generation, rendered session
requests, token grammar constraints, and stress generation on this host.

GPT-BigCode parity was added with `bigcode/tiny_starcoder_py`. The checkpoint
is 1.2 GB on disk, uses raw Transformers GPT-BigCode keys, and passed targeted
real-model generation, rendered session requests, token grammar constraints,
and stress generation on this host.

GPT-NeoX parity was added with `EleutherAI/pythia-70m-deduped`. The checkpoint
is 324 MB on disk, uses raw Transformers GPT-NeoX keys, and passed targeted
real-model generation, rendered session requests, token grammar constraints,
and stress generation on this host.

OLMo parity was added with `allenai/OLMo-1B-hf`. The checkpoint is 4.4 GB on
disk, uses raw Transformers OLMo keys, and passed targeted real-model
generation, rendered session requests, token grammar constraints, and stress
generation on this host.

GraniteMoE parity was added with
`ibm-granite/granite-3.1-1b-a400m-instruct`. The checkpoint is 2.5 GB on disk
and passed targeted real-model generation, rendered session requests, token
grammar constraints, and stress generation on this host.

DeepSeek parity was added with
`BlueMoonlight/deepseek-moe-16b-chat-mlx-4Bit`. The checkpoint is 8.6 GB on
disk and passed targeted release and serialized `main` real-model generation,
rendered session requests, token grammar constraints, and stress generation on
this host.

Mixtral parity was added with
`mlx-community/dolphin-2.9.1-mixtral-1x22b-2bit`. The checkpoint is 6.5 GB on
disk and passed targeted release and serialized `main` real-model generation,
rendered session requests, token grammar constraints, and stress generation on
this host. `mlx-community/Mixtral-SlimOrca-8x7B-3bit` is also present locally,
but the runtime memory guard rejects it on this 32 GB host before load because
the estimated 19.03 GB peak exceeds the active 16.64 GB guard ceiling.

Jamba parity was added with `mlx-community/AI21-Jamba-Reasoning-3B-4bit`. The
checkpoint is 1.6 GB on disk and passed targeted real-model generation,
rendered session requests, token grammar constraints, and stress generation on
this host.

EXAONE 3.x parity was added with
`mlx-community/EXAONE-3.5-2.4B-Instruct-4bit`. The checkpoint is 1.3 GB on disk
and passed targeted real-model generation, rendered session requests, token
grammar constraints, and stress generation on this host.

StableLM parity was added with `mlx-community/stablelm-2-zephyr-1_6b-4bit`.
The checkpoint is 1.1 GB on disk and passed targeted real-model generation,
rendered session requests, token grammar constraints, and stress generation on
this host.

Mamba parity was added with `mlx-community/mamba-130m-hf-bf16`. The checkpoint
is 246 MB on disk and passed targeted real-model generation, rendered session
requests, token grammar constraints, and stress generation on this host.

Mamba2 parity was added with `mlx-community/mamba2-130m-hf-4bit`. The
checkpoint is 70 MB on disk and passed targeted and serialized `main`
real-model generation, rendered session requests, token grammar constraints,
and stress generation on this host.

DeepSeek V2 parity was added with
`mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx`. The checkpoint is 8.2 GB on
disk and passed targeted and serialized `main` real-model generation, rendered
session requests, token grammar constraints, and stress generation on this
host.

Helium parity was added with `mlx-community/helium-1-preview-2b-4bit`. The
checkpoint is 1.1 GB on disk and passed targeted and serialized `main`
real-model generation, rendered session requests, token grammar constraints,
and stress generation on this host.

InternLM3 parity was added with `mlx-community/internlm3-8b-instruct-4bit`.
The checkpoint is 4.6 GB on disk and passed targeted release real-model
generation, rendered session requests, token grammar constraints, and stress
generation on this host.

Seed OSS parity was added with
`Open4bits/Seed-OSS-36B-Instruct-mlx-2Bit`. The checkpoint is 11 GB on disk and
passed targeted release generation, rendered session requests, token grammar
constraints, stress generation, the serialized `main` architecture sweep, and
the serialized `all` architecture sweep on this host.

The selected `deepseek-r1-distill-qwen-7b-4bit` checkpoint is a Qwen-distilled
model; its local `config.json` declares `model_type: qwen2`. The full
`deepseek-r1-4bit` checkpoint remains the true `deepseek_v3` coverage target,
but it is skipped on this host because it requires 256 GiB RAM.

## Benchmarks

These rows come from `BENCH` lines printed by the real-model test runner in
`.build/benchmarks/test-all-architectures-2026-06-25-independent-glm4moelite.log`.
They are short 8-token decode checks, so treat them as a regression snapshot
rather than a stable throughput claim.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | `qwen3-0.6b-4bit` | 8 | 16 | 0.0654 | 0.0306 | 0.0348 | 230.18 | 122.41 |
| `qwen3` | `qwen3-1.7b-4bit` | 8 | 16 | 0.1057 | 0.0432 | 0.0625 | 128.00 | 75.70 |
| `qwen3` | `qwen3-4b-4bit` | 8 | 16 | 0.1842 | 0.0672 | 0.1170 | 68.36 | 43.43 |
| `qwen3` | `qwen3-8b-4bit` | 8 | 17 | 0.2960 | 0.1139 | 0.1820 | 43.95 | 27.03 |
| `qwen2` | `qwen2.5-1.5b-4bit` | 8 | 37 | 0.0964 | 0.0391 | 0.0573 | 139.56 | 82.96 |
| `qwen2` | `qwen2.5-7b-4bit` | 8 | 37 | 0.2720 | 0.1057 | 0.1664 | 48.09 | 29.41 |
| `qwen2` | `qwen1.5-0.5b-chat-4bit` | 8 | 23 | 0.0482 | 0.0210 | 0.0272 | 294.15 | 166.15 |
| `llama` | `llama-3.2-1b-instruct-4bit` | 7 | 42 | 0.0751 | 0.0318 | 0.0433 | 161.82 | 93.25 |
| `llama` | `llama-3.2-3b-instruct-4bit` | 8 | 45 | 0.1484 | 0.0683 | 0.0800 | 99.98 | 53.92 |
| `llama` | `llama-3.1-8b-instruct-4bit` | 8 | 45 | 0.2931 | 0.1330 | 0.1602 | 49.95 | 27.29 |
| `llama` | `llama-3-8b-instruct-4bit` | 8 | 18 | 0.4240 | 0.2743 | 0.1498 | 53.42 | 18.87 |
| `mistral` | `mistral-7b-v0.3-4bit` | 8 | 10 | 0.2267 | 0.0823 | 0.1444 | 55.41 | 35.29 |
| `mistral` | `mistral-7b-v0.2-4bit` | 8 | 13 | 0.2229 | 0.0763 | 0.1466 | 54.58 | 35.89 |
| `mistral` | `mistral-nemo-2407-4bit` | 8 | 11 | 0.8622 | 0.6580 | 0.2042 | 39.17 | 9.28 |
| `phi` | `phi-2-hf-4bit-mlx` | 8 | 5 | 0.0992 | 0.0371 | 0.0621 | 128.74 | 80.61 |
| `phi3` | `phi-3.5-mini-instruct-4bit` | 8 | 8 | 0.1299 | 0.0392 | 0.0907 | 88.21 | 61.60 |
| `phi3` | `phi-4-mini-instruct-4bit` | 8 | 12 | 0.1676 | 0.0813 | 0.0863 | 92.70 | 47.72 |
| `gemma` | `gemma-2b-it-4bit` | 8 | 15 | 0.1218 | 0.0384 | 0.0834 | 95.91 | 65.68 |
| `gemma2` | `gemma-2-2b-it-4bit` | 8 | 14 | 0.1502 | 0.0796 | 0.0706 | 113.37 | 53.27 |
| `gemma2` | `gemma-2-9b-it-4bit` | 8 | 16 | 0.3075 | 0.1302 | 0.1773 | 45.12 | 26.01 |
| `gemma3` | `gemma-3-1b-it-qat-4bit` | 8 | 16 | 0.2433 | 0.1970 | 0.0463 | 172.78 | 32.88 |
| `gemma3` | `gemma-3-1b-it-4bit` | 8 | 18 | 0.2396 | 0.1938 | 0.0458 | 174.69 | 33.39 |
| `gemma3n` | `gemma-3n-e2b-it-lm-4bit` | 8 | 17 | 0.3908 | 0.2856 | 0.1052 | 76.06 | 20.47 |
| `gemma3n` | `gemma-3n-e4b-it-lm-4bit` | 8 | 17 | 0.4924 | 0.3379 | 0.1545 | 51.77 | 16.25 |
| `gemma4` | `gemma-4-e2b-it-4bit` | 8 | 19 | 0.2184 | 0.1149 | 0.1035 | 77.29 | 36.63 |
| `gemma4` | `gemma-4-e4b-it-4bit` | 8 | 21 | 0.2981 | 0.1642 | 0.1339 | 59.75 | 26.84 |
| `granite` | `granite-3.3-2b-instruct-4bit` | 8 | 65 | 0.1486 | 0.0600 | 0.0886 | 90.26 | 53.83 |
| `llama` | `smollm-135m-instruct-4bit` | 8 | 17 | 0.0461 | 0.0180 | 0.0281 | 284.48 | 173.64 |
| `smollm3` | `smollm3-3b-4bit` | 8 | 252 | 0.4743 | 0.1835 | 0.2908 | 27.51 | 16.87 |
| `lfm2` | `lfm2.5-1.2b-thinking-4bit` | 8 | 21 | 0.0731 | 0.0162 | 0.0569 | 140.60 | 109.42 |
| `lfm2_moe` | `lfm2-moe` | 8 | 17 | 0.1760 | 0.0859 | 0.0901 | 88.79 | 45.46 |
| `exaone` | `exaone-3.5-2.4b-instruct-4bit` | 8 | 43 | 0.1179 | 0.0336 | 0.0843 | 94.90 | 67.88 |
| `exaone4` | `exaone-4.0-1.2b-4bit` | 8 | 17 | 0.0990 | 0.0420 | 0.0571 | 140.17 | 80.78 |
| `ernie4_5` | `ernie-4.5-0.3b-bf16` | 8 | 11 | 0.0596 | 0.0151 | 0.0446 | 179.51 | 134.19 |
| `bitnet` | `bitnet-b1.58-2b-4t-4bit` | 8 | 10 | 0.1103 | 0.0344 | 0.0759 | 105.35 | 72.51 |
| `baichuan_m1` | `baichuan-m1-14b-instruct-4bit` | 8 | 10 | 0.7869 | 0.4965 | 0.2904 | 27.55 | 10.17 |
| `qwen2` | `deepseek-r1-distill-qwen-7b-4bit` | 8 | 10 | 0.2339 | 0.0937 | 0.1401 | 57.09 | 34.21 |
| `mimo` | `mimo-7b-rl-4bit` | 8 | 28 | 0.2889 | 0.1018 | 0.1871 | 42.76 | 27.70 |
| `glm4` | `glm-4-9b-0414-4bit` | 8 | 13 | 0.2746 | 0.1017 | 0.1729 | 46.26 | 29.13 |
| `acereason` | `acereason-nemotron-1.1-7b-4bit` | 8 | 33 | 0.3704 | 0.1471 | 0.2232 | 35.84 | 21.60 |
| `starcoder2` | `starcoder2-3b-4bit` | 8 | 6 | 0.1228 | 0.0452 | 0.0777 | 102.98 | 65.12 |
| `openelm` | `openelm-270m-instruct` | 8 | 5 | 0.0515 | 0.0112 | 0.0403 | 198.52 | 155.37 |
| `jamba` | `jamba-reasoning-3b-4bit` | 8 | 46 | 0.2802 | 0.1963 | 0.0839 | 95.37 | 28.55 |
| `internlm2` | `internlm2.5-7b-chat-4bit` | 8 | 18 | 0.2115 | 0.0700 | 0.1415 | 56.52 | 37.82 |
| `falcon_h1` | `falcon-h1-0.5b-instruct-4bit` | 8 | 19 | 0.0965 | 0.0459 | 0.0506 | 158.14 | 82.91 |
| `qwen3_5` | `qwen3.5` | 8 | 18 | 0.2109 | 0.0963 | 0.1146 | 69.81 | 37.94 |
| `olmo3` | `olmo3` | 8 | 85 | 0.4827 | 0.2317 | 0.2510 | 31.87 | 16.57 |
| `apertus` | `apertus` | 8 | 76 | 0.5640 | 0.3200 | 0.2440 | 32.78 | 14.18 |

## Qwen2 MoE Parity Check

These rows come from the Qwen2 MoE runs on 2026-06-25. The 8-token row is the
targeted architecture check; the 4-token row is from the serialized
`main` architecture sweep.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2_moe` | `qwen1.5-moe-a2.7b-chat-4bit` | 8 | 29 | 0.7086 | 0.6075 | 0.1011 | 79.14 | 11.29 |
| `qwen2_moe` | `qwen1.5-moe-a2.7b-chat-4bit` | 4 | 29 | 0.1902 | 0.0991 | 0.0911 | 43.92 | 21.04 |

## Qwen Parity Check

These rows come from the targeted release Qwen run and the serialized debug
`main` sweep on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen` | `qwen-1.8b` release targeted | 4 | 17 | 0.1085 | 0.0574 | 0.0510 | 78.38 | 36.88 |
| `qwen` | `qwen-1.8b` debug main | 2 | 17 | 0.0814 | 0.0493 | 0.0321 | 62.24 | 24.57 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen` | `qwen-1.8b` release targeted | 32 | 0.3389 | 0.0226 | 0.3163 | 101.17 | 94.41 |
| `qwen` | `qwen-1.8b` debug main | 32 | 0.4424 | 0.0260 | 0.4164 | 76.86 | 72.33 |

## GLM Parity Check

These rows come from the targeted release GLM run and the serialized release
`main` sweep on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `glm` | `glm-edge-1.5b-chat` targeted | 4 | 14 | 0.0921 | 0.0429 | 0.0492 | 81.37 | 43.44 |
| `glm` | `glm-edge-1.5b-chat` main | 4 | 14 | 0.0886 | 0.0394 | 0.0493 | 81.19 | 45.13 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `glm` | `glm-edge-1.5b-chat` targeted | 32 | 0.3393 | 0.0238 | 0.3154 | 101.44 | 94.32 |
| `glm` | `glm-edge-1.5b-chat` main | 32 | 0.3375 | 0.0234 | 0.3141 | 101.88 | 94.81 |

## Hunyuan V1 Dense Parity Check

These rows come from the targeted release Hunyuan MT run and the serialized
release `main` sweep on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hunyuan_v1_dense` | `hunyuan-mt-7b` targeted | 4 | 12 | 2.6591 | 2.2062 | 0.4529 | 8.83 | 1.50 |
| `hunyuan_v1_dense` | `hunyuan-mt-7b` main | 4 | 12 | 2.0707 | 1.7768 | 0.2940 | 13.61 | 1.93 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hunyuan_v1_dense` | `hunyuan-mt-7b` targeted | 32 | 1.5269 | 0.1150 | 1.4119 | 22.66 | 20.96 |
| `hunyuan_v1_dense` | `hunyuan-mt-7b` main | 32 | 1.5290 | 0.1149 | 1.4142 | 22.63 | 20.93 |

## Phixtral Parity Check

These rows come from the targeted release Phixtral run and the serialized debug
`main` sweep on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phixtral` | `phixtral-2x2_8` release targeted | 4 | 23 | 0.5923 | 0.4387 | 0.1536 | 26.04 | 6.75 |
| `phixtral` | `phixtral-2x2_8` debug main | 4 | 23 | 0.5458 | 0.2573 | 0.2885 | 13.86 | 7.33 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phixtral` | `phixtral-2x2_8` release targeted | 32 | 0.9589 | 0.0962 | 0.8627 | 37.09 | 33.37 |
| `phixtral` | `phixtral-2x2_8` debug main | 32 | 1.2686 | 0.1056 | 1.1630 | 27.52 | 25.23 |

## Cohere2 Parity Check

These rows come from the targeted release Cohere2 run and the serialized debug
`main` and release `all` sweeps on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cohere2` | `cohere2-tiny-aya-global-4bit` release targeted | 4 | 392 | 0.6703 | 0.3593 | 0.3110 | 12.86 | 5.97 |
| `cohere2` | `cohere2-tiny-aya-global-4bit` debug main | 4 | 392 | 0.6517 | 0.3130 | 0.3387 | 11.81 | 6.14 |
| `cohere2` | `cohere2-tiny-aya-global-4bit` release all | 4 | 392 | 0.6170 | 0.3144 | 0.3026 | 13.22 | 6.48 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cohere2` | `cohere2-tiny-aya-global-4bit` release targeted | 28 | 0.8007 | 0.2961 | 0.5046 | 55.48 | 34.97 |
| `cohere2` | `cohere2-tiny-aya-global-4bit` debug main | 28 | 1.0836 | 0.3005 | 0.7830 | 35.76 | 25.84 |
| `cohere2` | `cohere2-tiny-aya-global-4bit` release all | 28 | 0.7971 | 0.2951 | 0.5020 | 55.78 | 35.13 |

## RWKV7 Parity Check

These rows come from the targeted release RWKV7 run and the serialized debug
`main` and release `all` sweeps on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rwkv7` | `rwkv7-g1d-0.1b` release targeted | 4 | 9 | 0.4956 | 0.4546 | 0.0409 | 97.71 | 8.07 |
| `rwkv7` | `rwkv7-g1d-0.1b` debug main | 4 | 9 | 0.1138 | 0.0370 | 0.0769 | 52.04 | 35.14 |
| `rwkv7` | `rwkv7-g1d-0.1b` release all | 4 | 9 | 0.0379 | 0.0199 | 0.0180 | 222.30 | 105.44 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rwkv7` | `rwkv7-g1d-0.1b` release targeted | 32 | 0.1023 | 0.0059 | 0.0964 | 332.02 | 312.80 |
| `rwkv7` | `rwkv7-g1d-0.1b` debug main | 32 | 0.6170 | 0.0249 | 0.5922 | 54.04 | 51.86 |
| `rwkv7` | `rwkv7-g1d-0.1b` release all | 32 | 0.1113 | 0.0059 | 0.1053 | 303.77 | 287.61 |

## MiniCPM3 Parity Check

This row comes from the targeted MiniCPM3 real-model run on 2026-06-25.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `minicpm3` | `minicpm3-4b-4bit` | 8 | 18 | 0.2570 | 0.0973 | 0.1597 | 50.09 | 31.13 |

## GPT-2 Parity Check

These rows come from the targeted GPT-2 real-model run on 2026-06-25.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | `gpt2-base-mlx` | 8 | 5 | 0.3261 | 0.2633 | 0.0627 | 127.52 | 24.54 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | `gpt2-base-mlx` | 32 | 0.1829 | 0.0083 | 0.1745 | 183.35 | 174.98 |

## GPT-BigCode Parity Check

These rows come from the targeted GPT-BigCode real-model run on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt_bigcode` | `tiny-starcoder-py` | 8 | 5 | 0.2235 | 0.1369 | 0.0866 | 92.37 | 35.80 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt_bigcode` | `tiny-starcoder-py` | 32 | 0.3118 | 0.0130 | 0.2988 | 107.08 | 102.63 |

## GPT-NeoX Parity Check

These rows come from the targeted GPT-NeoX real-model run on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt_neox` | `pythia-70m-deduped` | 8 | 5 | 0.3738 | 0.3311 | 0.0427 | 187.39 | 21.40 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt_neox` | `pythia-70m-deduped` | 32 | 0.1010 | 0.0051 | 0.0959 | 333.66 | 316.77 |

## OLMo Parity Check

These rows come from the targeted OLMo real-model run on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `olmo` | `olmo-1b-hf` | 8 | 10 | 0.3873 | 0.2479 | 0.1394 | 57.39 | 20.66 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `olmo` | `olmo-1b-hf` | 32 | 0.4752 | 0.0171 | 0.4580 | 69.86 | 67.34 |

## Jamba Parity Check

These rows come from targeted and serialized `main` Jamba real-model runs on
2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `jamba` | `jamba-reasoning-3b-4bit` | 8 | 46 | 0.2802 | 0.1963 | 0.0839 | 95.37 | 28.55 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jamba` | `jamba-reasoning-3b-4bit` | 32 | 0.5202 | 0.2435 | 0.2767 | 115.65 | 61.52 |

## StableLM Parity Check

These rows come from the targeted StableLM real-model run on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stablelm` | `stablelm-2-zephyr-1.6b-4bit` | 8 | 23 | 0.1651 | 0.0447 | 0.1205 | 66.40 | 48.45 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stablelm` | `stablelm-2-zephyr-1.6b-4bit` | 32 | 0.5232 | 0.0463 | 0.4769 | 67.10 | 61.16 |

## Mamba Parity Check

These rows come from the serialized `main` Mamba real-model run on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mamba` | `mamba-130m-hf-bf16` | 8 | 5 | 0.0507 | 0.0258 | 0.0249 | 321.04 | 157.84 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mamba` | `mamba-130m-hf-bf16` | 32 | 0.0933 | 0.0144 | 0.0790 | 405.31 | 342.84 |

## Mamba2 Parity Check

These rows come from the serialized `main` Mamba2 real-model run on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mamba2` | `mamba2-130m-hf-4bit` | 8 | 5 | 0.0455 | 0.0211 | 0.0244 | 328.15 | 175.99 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mamba2` | `mamba2-130m-hf-4bit` | 32 | 0.0658 | 0.0051 | 0.0607 | 527.55 | 486.27 |

## Helium Parity Check

These rows come from the serialized `main` sweep on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `helium` | `helium-1-preview-2b-4bit` | 8 | 5 | 0.0826 | 0.0141 | 0.0685 | 116.79 | 96.86 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `helium` | `helium-1-preview-2b-4bit` | 32 | 0.1868 | 0.0050 | 0.1819 | 175.94 | 171.27 |

## DeepSeek V2 Parity Check

These rows come from the serialized `main` DeepSeek V2 Lite real-model run on
2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek_v2` | `deepseek-v2-lite-chat-4bit` | 2 | 16 | 0.3533 | 0.2940 | 0.0593 | 33.71 | 5.66 |

Best stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek_v2` | `deepseek-v2-lite-chat-4bit` | 32 | 0.3656 | 0.0988 | 0.2669 | 119.91 | 87.52 |

## DeepSeek Parity Check

These rows come from the targeted release DeepSeek MoE run and the serialized
debug `main` sweep on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek` | `deepseek-moe-16b-chat-4bit` release targeted | 2 | 16 | 0.7621 | 0.7169 | 0.0451 | 44.32 | 2.62 |
| `deepseek` | `deepseek-moe-16b-chat-4bit` debug main | 2 | 16 | 0.2262 | 0.1594 | 0.0668 | 29.94 | 8.84 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek` | `deepseek-moe-16b-chat-4bit` release targeted | 32 | 0.4319 | 0.1276 | 0.3043 | 105.15 | 74.09 |
| `deepseek` | `deepseek-moe-16b-chat-4bit` debug main | 32 | 0.9222 | 0.1447 | 0.7775 | 41.16 | 34.70 |

## Phi-3-small Parity Check

These rows come from the targeted release Phi-3-small run on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi3small` | `phi-3-small-8k-instruct-aq4-32` | 4 | 16 | 0.2060 | 0.1118 | 0.0942 | 42.47 | 19.41 |

Best decode stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi3small` | `phi-3-small-8k-instruct-aq4-32` | 32 | 0.7706 | 0.1575 | 0.6132 | 52.19 | 41.52 |

## InternLM3 Parity Check

These rows come from the targeted release InternLM3 run on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `internlm3` | `internlm3-8b-instruct-4bit` | 4 | 19 | 0.1988 | 0.0936 | 0.1052 | 38.02 | 20.12 |

Best decode stress iteration from the same run:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `internlm3` | `internlm3-8b-instruct-4bit` | 32 | 0.6669 | 0.1146 | 0.5523 | 57.94 | 47.98 |

## Seed OSS Parity Check

These rows come from the targeted release Seed OSS run and the serialized debug
`all` sweep on 2026-07-01.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed_oss` | `seed-oss-36b-instruct-2bit` release targeted | 4 | 18 | 2.1351 | 1.7696 | 0.3655 | 10.94 | 1.87 |
| `seed_oss` | `seed-oss-36b-instruct-2bit` debug all | 2 | 18 | 1.1444 | 0.9221 | 0.2223 | 8.99 | 1.75 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed_oss` | `seed-oss-36b-instruct-2bit` release targeted | 32 | 3.5508 | 1.5092 | 2.0416 | 15.67 | 9.01 |
| `seed_oss` | `seed-oss-36b-instruct-2bit` debug all | 32 | 3.3109 | 1.1842 | 2.1267 | 15.05 | 9.66 |

## Mixtral Parity Check

These rows come from the targeted release Mixtral run and the serialized debug
`main` sweep on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mixtral` | `mixtral-1x22b-2bit` release targeted | 2 | 20 | 0.5520 | 0.3806 | 0.1714 | 11.67 | 3.62 |
| `mixtral` | `mixtral-1x22b-2bit` debug main | 2 | 20 | 0.6011 | 0.3999 | 0.2012 | 9.94 | 3.33 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mixtral` | `mixtral-1x22b-2bit` release targeted | 32 | 1.8707 | 0.6612 | 1.2095 | 26.46 | 17.11 |
| `mixtral` | `mixtral-1x22b-2bit` debug main | 32 | 2.1413 | 0.6688 | 1.4725 | 21.73 | 14.94 |

## GraniteMoE Parity Check

These rows come from the targeted release GraniteMoE real-model run and the
serialized debug `main` sweep on 2026-06-30.

| Architecture | Model | Generated | Prompt | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `granitemoe` | `granite-3.1-1b-a400m-instruct` release targeted | 4 | 67 | 0.4020 | 0.3575 | 0.0445 | 89.95 | 9.95 |
| `granitemoe` | `granite-3.1-1b-a400m-instruct` debug main | 2 | 67 | 0.1284 | 0.0709 | 0.0574 | 34.82 | 15.58 |

Best stress iterations from the same runs:

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `granitemoe` | `granite-3.1-1b-a400m-instruct` release targeted | 32 | 0.1844 | 0.0204 | 0.1641 | 195.02 | 173.49 |
| `granitemoe` | `granite-3.1-1b-a400m-instruct` debug main | 32 | 0.8046 | 0.0346 | 0.7699 | 41.56 | 39.77 |

## Small-Fit Stress Baseline

These rows are the best warm iterations from
`.build/benchmarks/test-all-architectures-2026-06-25-small-fit.log`. The stress
test is intentionally serialized and keeps each model loaded for the repeated
generations.

| Architecture | Model | Generated | Total s | Prompt s | Decode s | Decode tok/s | E2E tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3_moe` | `qwen3-moe-30b-a3b-3bit` | 32 | 0.6304 | 0.1767 | 0.4537 | 70.53 | 50.76 |
| `mistral` | `mistral-small-24b-2501-3bit` | 18 | 0.8913 | 0.2025 | 0.6888 | 26.13 | 20.20 |
| `gpt_oss` | `gpt-oss-20b-mxfp4-jinx` | 32 | 1.5133 | 0.2260 | 1.2873 | 24.86 | 21.15 |
| `qwen3_5_moe` | `qwen3.5-moe-35b-a3b-3bit` | 32 | 0.5631 | 0.1449 | 0.4182 | 76.52 | 56.83 |
| `nemotron_h` | `nemotron-h-nano-4b-4bit` | 32 | 0.4026 | 0.0621 | 0.3405 | 93.99 | 79.48 |

## Skipped By Memory Gate

Presence is for the exact skipped model, not smaller siblings or different
quantization levels.

| Model | Reason | In `.build/test-models` | Other local copy found |
| --- | --- | --- | --- |
| `qwen3-moe-30b-a3b-4bit` | Requires 48 GiB RAM. | No. | No. |
| `mistral-small-24b-2501-4bit` | Requires 48 GiB RAM. | No. | No. |
| `phi-3.5-moe-instruct-4bit` | Requires 48 GiB RAM. | No. | No exact copy found in targeted checks on this host. |
| `gemma-3n-e2b-it-lm-bf16` | Requires 48 GiB RAM. | No. | No; only the 4-bit sibling is present. |
| `deepseek-r1-4bit` | Requires 256 GiB RAM. | No. | No. |
| `cohere-command-r-v01-4bit` | Estimated runtime memory about 53 GiB; host budget is 24 GiB. | Yes, `c4ai-command-r-v01-4bit`. | Yes, Patagonia/Think MLXSession resource copies. |
| `gpt-oss` | Requires 48 GiB RAM. | No. | No exact copy; `gpt-oss-20b-mxfp4-jinx` is the 32 GB alternate. |
| `qwen3-next` | Requires 64 GiB RAM. | No. | No. |
| `qwen3.5-moe` | Requires 48 GiB RAM. | No. | No exact copy; `qwen3.5-moe-35b-a3b-3bit` is the 32 GB alternate. |
