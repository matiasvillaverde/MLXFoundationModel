#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATALOG_PATH="${MLX_MODEL_CATALOG_PATH:-$ROOT_DIR/Tests/MLXRealModelTests/Resources/model-catalog.json}"
SUMMARY_SCRIPT="$ROOT_DIR/scripts/summarize-real-model-run.py"
MODEL_DIR="${MLX_TEST_MODELS_DIR:-$ROOT_DIR/.models}"
SCOPE="${MLX_REAL_MODEL_SCOPE:-relevant}"
CONFIGURATION="${CONFIGURATION:-debug}"
MODEL_TIMEOUT_SECONDS="${MLX_REAL_MODEL_TIMEOUT_SECONDS:-1200}"
FEATURE_TIMEOUT_SECONDS="${MLX_REAL_MODEL_FEATURE_TIMEOUT_SECONDS:-900}"
GENERATION_TOKENS="${MLX_REAL_MODEL_GENERATION_TOKENS:-2}"
GENERATION_TIMEOUT_SECONDS="${MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS:-120}"
ALLOW_OVERSIZED_MODELS="${MLX_ALLOW_OVERSIZED_MODELS:-0}"
SKIP_BUILD_AFTER_FIRST="${MLX_REAL_MODEL_SKIP_BUILD_AFTER_FIRST:-1}"
MODEL_STORAGE_TIMEOUT_SECONDS="${MLX_MODEL_STORAGE_TIMEOUT_SECONDS:-10}"
MEMORY_GUARD_TIER="${MLX_REAL_MODEL_MEMORY_GUARD_TIER:-}"
MEMORY_GUARD_HARD_LIMIT_FRACTION="${MLX_REAL_MODEL_MEMORY_GUARD_HARD_LIMIT_FRACTION:-}"
DRY_RUN="${MLX_REAL_MODEL_DRY_RUN:-0}"
BENCHMARK_DIR="${MLX_REAL_MODEL_BENCHMARK_DIR:-$ROOT_DIR/.build/benchmarks}"
BENCHMARK_LOG="${MLX_REAL_MODEL_BENCHMARK_LOG:-$BENCHMARK_DIR/real-models-$(date -u +%Y%m%dT%H%M%SZ).log}"
BENCHMARK_SUMMARY="${MLX_REAL_MODEL_BENCHMARK_SUMMARY:-${BENCHMARK_LOG%.log}-summary.json}"
RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

run_with_timeout() {
  local seconds="$1"
  shift
  python3 - "$seconds" "$@" <<'PY'
import subprocess
import sys

seconds = int(sys.argv[1])
command = sys.argv[2:]
process = subprocess.Popen(command)
try:
    sys.exit(process.wait(timeout=seconds))
except subprocess.TimeoutExpired:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    print(
        f"Timed out after {seconds}s: {' '.join(command)}",
        file=sys.stderr,
    )
    sys.exit(124)
PY
}

check_model_storage() {
  python3 - "$MODEL_DIR" "$MODEL_STORAGE_TIMEOUT_SECONDS" <<'PY'
import os
import signal
import sys

model_dir = sys.argv[1]
timeout_seconds = int(sys.argv[2])


def timeout_handler(_signum, _frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(timeout_seconds)

try:
    if not os.path.isdir(model_dir):
        print(
            f"Model directory does not exist: {model_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    with os.scandir(model_dir) as entries:
        next(entries, None)
except TimeoutError:
    print(
        f"Model directory did not respond within {timeout_seconds}s: {model_dir}",
        file=sys.stderr,
    )
    print(
        "Check the model volume or set MLX_TEST_MODELS_DIR to a responsive model root.",
        file=sys.stderr,
    )
    sys.exit(124)
except OSError as error:
    print(f"Cannot inspect model directory {model_dir}: {error}", file=sys.stderr)
    sys.exit(1)
finally:
    signal.alarm(0)
PY
}

host_memory_gb() {
  python3 <<'PY'
import os
import platform
import subprocess

bytes_value = 0
if platform.system() == "Darwin":
    try:
        bytes_value = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        bytes_value = 0
elif os.path.exists("/proc/meminfo"):
    with open("/proc/meminfo", "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("MemTotal:"):
                bytes_value = int(line.split()[1]) * 1024
                break

print(max(1, bytes_value // (1024 ** 3)))
PY
}

HOST_MEMORY_GB="${MLX_HOST_MEMORY_GB:-$(host_memory_gb)}"
check_model_storage

mapfile -t MODELS < <(
  python3 - "$CATALOG_PATH" "$SCOPE" "$MODEL_DIR" "$HOST_MEMORY_GB" "$ALLOW_OVERSIZED_MODELS" <<'PY'
import json
import os
import sys
from pathlib import Path

catalog_path = Path(sys.argv[1])
scope = sys.argv[2]
model_dir = Path(sys.argv[3])
host_memory_gb = int(sys.argv[4])
allow_oversized_models = sys.argv[5] == "1"
selected_ids = {
    value.strip()
    for value in os.environ.get("MLX_REAL_MODEL_IDS", "").split(",")
    if value.strip()
}

with catalog_path.open("r", encoding="utf-8") as file:
    models = json.load(file)

ONE_GIB = 1024 ** 3
MODEL_LOAD_OVERHEAD_MULTIPLIER = 2.5
MODEL_ARTIFACT_EXTENSIONS = {".bin", ".mlx", ".npz", ".safetensors"}


def has_model_files(model):
    path = model_dir / model["relativePath"]
    has_config = (path / "config.json").exists()
    has_tokenizer = (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or (path / "cl100k_base.tiktoken").exists()
        or (path / "tiktoken.model").exists()
        or (path / "hy.tiktoken").exists()
        or (path / "qwen.tiktoken").exists()
    )
    has_weights = (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or any(path.glob("*.safetensors"))
    )
    return has_config and has_tokenizer and has_weights


def estimated_model_load_bytes(model):
    path = model_dir / model["relativePath"]
    if not path.exists():
        return None
    total = 0
    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in MODEL_ARTIFACT_EXTENSIONS:
            total += file.stat().st_size
    return total or None


def host_model_budget_bytes():
    reserve_gb = 4 if host_memory_gb < 24 else 8
    return max(1, host_memory_gb - reserve_gb) * ONE_GIB


def estimated_runtime_bytes(model_load_bytes):
    return int((model_load_bytes * MODEL_LOAD_OVERHEAD_MULTIPLIER) + 0.999999)


def can_run_within_host_memory(model):
    minimum_memory_gb = model.get("minimumMemoryGB")
    if minimum_memory_gb is not None:
        if minimum_memory_gb <= host_memory_gb:
            return True
        print(
            "Skipping "
            f"{model['id']}: requires {minimum_memory_gb} GiB RAM, "
            f"host has {host_memory_gb} GiB. "
            "Set MLX_ALLOW_OVERSIZED_MODELS=1 to run anyway.",
            file=sys.stderr,
        )
        return False

    model_load_bytes = estimated_model_load_bytes(model)
    if model_load_bytes is None:
        return True

    runtime_bytes = estimated_runtime_bytes(model_load_bytes)
    if runtime_bytes <= host_model_budget_bytes():
        return True

    runtime_gb = int((runtime_bytes + ONE_GIB - 1) // ONE_GIB)
    budget_gb = int(host_model_budget_bytes() // ONE_GIB)
    print(
        "Skipping "
        f"{model['id']}: local weights need about {runtime_gb} GiB runtime memory, "
        f"host budget is {budget_gb} GiB. "
        "Set MLX_ALLOW_OVERSIZED_MODELS=1 to run anyway.",
        file=sys.stderr,
    )
    return False


def is_selected(model):
    if not model.get("repository"):
        return False
    if selected_ids:
        return model["id"] in selected_ids
    if scope == "all":
        return True
    if scope == "downloaded":
        return has_model_files(model)
    if scope in {"main", "relevant", "smoke"}:
        return scope in model.get("tags", [])
    fields = [
        model["id"],
        model["displayName"],
        model["architecture"],
        model.get("repository", ""),
        model["relativePath"],
        ",".join(model.get("tags", [])),
    ]
    return scope.lower() in " ".join(fields).lower()


for model in models:
    if not is_selected(model):
        continue
    if not allow_oversized_models and not can_run_within_host_memory(model):
        continue
    print(
        "\t".join(
            [
                model["id"],
                model["displayName"],
                model["architecture"],
                ",".join(model.get("tags", [])),
            ]
        )
    )
PY
)

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No downloadable models matched scope '$SCOPE'."
  exit 0
fi

echo "MLXFoundationModel per-model real validation"
echo "Catalog:               $CATALOG_PATH"
echo "Models:                $MODEL_DIR"
echo "Scope:                 $SCOPE"
echo "Host RAM:              ${HOST_MEMORY_GB} GiB"
echo "Selected model count:  ${#MODELS[@]}"
echo "Per-model timeout:     ${MODEL_TIMEOUT_SECONDS}s"
echo "Feature timeout:       ${FEATURE_TIMEOUT_SECONDS}s"
echo "Storage timeout:       ${MODEL_STORAGE_TIMEOUT_SECONDS}s"
echo "Generation token cap:  ${GENERATION_TOKENS}"
echo "Memory guard tier:     ${MEMORY_GUARD_TIER:-catalog/default}"
echo "Skip rebuild checks:   $([[ "$SKIP_BUILD_AFTER_FIRST" == "1" ]] && echo "enabled" || echo "disabled")"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run:               enabled"
fi
echo

if [[ "$DRY_RUN" == "1" ]]; then
  COUNT=0
  for MODEL in "${MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS=$'\t' read -r ID DISPLAY_NAME ARCHITECTURE TAGS <<<"$MODEL"
    echo "[$COUNT/${#MODELS[@]}] $DISPLAY_NAME"
    echo "   id:           $ID"
    echo "   architecture: $ARCHITECTURE"
    echo "   tags:         $TAGS"
  done
  exit 0
fi

mkdir -p "$(dirname "$BENCHMARK_LOG")" "$(dirname "$BENCHMARK_SUMMARY")"
{
  echo "MLXFoundationModel real-model validation"
  echo "started_utc: $RUN_STARTED_UTC"
  echo "catalog: $CATALOG_PATH"
  echo "models: $MODEL_DIR"
  echo "scope: $SCOPE"
  echo "host_ram_gib: $HOST_MEMORY_GB"
  echo "selected_model_count: ${#MODELS[@]}"
  echo
} >"$BENCHMARK_LOG"
echo "Benchmark log:         $BENCHMARK_LOG"
echo "Benchmark summary:     $BENCHMARK_SUMMARY"
echo

log_benchmark_line() {
  printf '%s\n' "$*" | tee -a "$BENCHMARK_LOG"
}

lookup_model_metadata() {
  local expected_id="$1"
  local model
  local id
  local display_name
  local architecture
  local tags

  for model in "${MODELS[@]}"; do
    IFS=$'\t' read -r id display_name architecture tags <<<"$model"
    if [[ "$id" == "$expected_id" ]]; then
      printf '%s\t%s\n' "$architecture" "$tags"
      return 0
    fi
  done
  printf '\t\n'
}

test_metadata_json() {
  local label="$1"
  local model_id="$2"
  local feature_key="$3"
  local architecture="$4"
  local tags="$5"

  python3 - "$label" "$model_id" "$feature_key" "$architecture" "$tags" <<'PY'
import json
import sys

label, model_id, feature_key, architecture, tags = sys.argv[1:]
feature_keys = [value.strip() for value in feature_key.split(",") if value.strip()]
record = {
    "schema_version": 1,
    "label": label,
    "model_id": model_id or None,
    "feature_key": feature_keys[0] if feature_keys else None,
    "feature_keys": feature_keys,
    "architecture": architecture or None,
    "tags": [value for value in tags.split(",") if value],
}
print(json.dumps(record, sort_keys=True))
PY
}

log_test_start() {
  local label="$1"
  local model_id="$2"
  local feature_key="$3"
  local architecture="$4"
  local tags="$5"
  local metadata_json

  log_benchmark_line "-> $label"
  metadata_json="$(test_metadata_json "$label" "$model_id" "$feature_key" "$architecture" "$tags")"
  log_benchmark_line "TEST_META_JSON $metadata_json"
}

write_benchmark_summary() {
  local result="$1"
  python3 "$SUMMARY_SCRIPT" \
    --log "$BENCHMARK_LOG" \
    --summary "$BENCHMARK_SUMMARY" \
    --result "$result" \
    --started-utc "$RUN_STARTED_UTC" \
    --catalog "$CATALOG_PATH" \
    --models "$MODEL_DIR" \
    --scope "$SCOPE" \
    --host-memory-gb "$HOST_MEMORY_GB" \
    --selected-count "${#MODELS[@]}" \
    --swift-test-invocation-count "$SWIFT_TEST_INVOCATION_COUNT"
  log_benchmark_line "Benchmark summary: $BENCHMARK_SUMMARY"
}

SWIFT_TEST_FLAGS=(--configuration "$CONFIGURATION" --no-parallel)
COMMON_ENV=(
  MLX_RUN_REAL_MODEL_TESTS=1
  MLX_REAL_MODEL_SCOPE="$SCOPE"
  MLX_REAL_MODEL_GENERATION_TOKENS="$GENERATION_TOKENS"
  MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS="$GENERATION_TIMEOUT_SECONDS"
  MLX_HOST_MEMORY_GB="$HOST_MEMORY_GB"
  MLX_ALLOW_OVERSIZED_MODELS="$ALLOW_OVERSIZED_MODELS"
)
if [[ -n "$MEMORY_GUARD_TIER" ]]; then
  COMMON_ENV+=(MLX_REAL_MODEL_MEMORY_GUARD_TIER="$MEMORY_GUARD_TIER")
fi
if [[ -n "$MEMORY_GUARD_HARD_LIMIT_FRACTION" ]]; then
  COMMON_ENV+=(MLX_REAL_MODEL_MEMORY_GUARD_HARD_LIMIT_FRACTION="$MEMORY_GUARD_HARD_LIMIT_FRACTION")
fi
FAILURES=()
LAST_TEST_STATUS=0
SWIFT_TEST_INVOCATION_COUNT=0

run_swift_test() {
  local label="$1"
  local timeout_seconds="$2"
  local filter="$3"
  local model_id="${4:-}"
  local feature_key="${5:-}"
  local architecture=""
  local tags=""
  local metadata=""
  local environment=("${COMMON_ENV[@]}")
  local swift_test_flags=("${SWIFT_TEST_FLAGS[@]}")
  if [[ -n "$model_id" ]]; then
    environment+=(MLX_REAL_MODEL_IDS="$model_id")
    metadata="$(lookup_model_metadata "$model_id")"
    IFS=$'\t' read -r architecture tags <<<"$metadata"
  fi
  if [[ "$SKIP_BUILD_AFTER_FIRST" == "1" && "$SWIFT_TEST_INVOCATION_COUNT" -gt 0 ]]; then
    swift_test_flags+=(--skip-build)
  fi

  local started_seconds=$SECONDS
  log_test_start "$label" "$model_id" "$feature_key" "$architecture" "$tags"
  if run_with_timeout "$timeout_seconds" \
    env "${environment[@]}" swift test "${swift_test_flags[@]}" --filter "$filter" \
    2>&1 | tee -a "$BENCHMARK_LOG"; then
    local duration_seconds=$((SECONDS - started_seconds))
    log_benchmark_line "   passed"
    log_benchmark_line "   duration_seconds: $duration_seconds"
    LAST_TEST_STATUS=0
    SWIFT_TEST_INVOCATION_COUNT=$((SWIFT_TEST_INVOCATION_COUNT + 1))
    return 0
  else
    local status=$?
    local duration_seconds=$((SECONDS - started_seconds))
    log_benchmark_line "   failed with exit status $status"
    log_benchmark_line "   duration_seconds: $duration_seconds"
    FAILURES+=("$label (exit status $status)")
    LAST_TEST_STATUS="$status"
    SWIFT_TEST_INVOCATION_COUNT=$((SWIFT_TEST_INVOCATION_COUNT + 1))
    return 0
  fi
}

model_is_selected() {
  local expected_id="$1"
  local model
  local id
  local display_name
  local architecture
  local tags

  for model in "${MODELS[@]}"; do
    IFS=$'\t' read -r id display_name architecture tags <<<"$model"
    if [[ "$id" == "$expected_id" ]]; then
      return 0
    fi
  done
  return 1
}

run_model_feature_test() {
  local model_id="$1"
  local label="$2"
  local timeout_seconds="$3"
  local filter="$4"
  local feature_key="$5"

  if model_is_selected "$model_id"; then
    run_swift_test "$label" "$timeout_seconds" "$filter" "$model_id" "$feature_key"
  else
    log_test_start "$label" "$model_id" "$feature_key" "" ""
    log_benchmark_line "   skipped: $model_id is not selected for scope '$SCOPE'"
    log_benchmark_line "   duration_seconds: 0"
  fi
}

coverage_passed() {
  python3 - "$BENCHMARK_SUMMARY" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    summary = json.load(file)

coverage = summary.get("feature_coverage", {})
sys.exit(0 if coverage.get("passed") is True else 1)
PY
}

benchmark_coverage_passed() {
  python3 - "$BENCHMARK_SUMMARY" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    summary = json.load(file)

coverage = summary.get("benchmark_coverage", {})
sys.exit(0 if coverage.get("passed") is True else 1)
PY
}

architecture_supports_native_tool_constraints() {
  case "$1" in
    cohere|cohere2|deepseek_v4|function_gemma|gemma|gemma2|gemma3|gemma3_text|gemma3n|gemma4|gemma4_text) return 0 ;;
    glm|glm4|glm4_moe|glm4_moe_lite|glm_moe_dsa|gpt_oss|kimi_k2|kimi_k25) return 0 ;;
    longcat_flash|longcat_flash_ngram|minimax|minimax_m3|mistral|mistral3|mixtral) return 0 ;;
    qwen|qwen2|qwen2_moe|qwen3|qwen3_5|qwen3_5_moe|qwen3_moe|qwen3_next) return 0 ;;
    *) return 1 ;;
  esac
}

log_coverage_failures() {
  python3 - "$BENCHMARK_SUMMARY" <<'PY' | tee -a "$BENCHMARK_LOG"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    summary = json.load(file)

rows = summary.get("feature_coverage", {}).get("rows", [])
for row in rows:
    if row.get("status") == "passed":
        continue
    model_id = row.get("model_id") or "<unknown>"
    feature_key = row.get("feature_key") or "<unknown>"
    status = row.get("status") or "<unknown>"
    print(f"- {model_id} {feature_key}: {status}")
PY
}

log_benchmark_coverage_failures() {
  python3 - "$BENCHMARK_SUMMARY" <<'PY' | tee -a "$BENCHMARK_LOG"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    summary = json.load(file)

rows = summary.get("benchmark_coverage", {}).get("rows", [])
for row in rows:
    if row.get("status") == "passed":
        continue
    model_id = row.get("model_id") or "<unknown>"
    status = row.get("status") or "<unknown>"
    print(f"- {model_id}: {status}")
PY
}

run_swift_test \
  "catalog metadata" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelCatalogTests" \
  "" \
  "catalog_metadata"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 sampling and logits controls" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelSamplingTests/qwen3GenerationAppliesSamplingAndLogitsControls" \
  "sampling_logits"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 rendered text streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelInterfaceTests/qwen3StreamsMultipleChunksFromRenderedTextRequest" \
  "rendered_text_streaming"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 stream lifecycle events" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelInterfaceTests/qwen3RawStreamReportsLifecyclePhaseBoundaries" \
  "stream_lifecycle"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 on-demand stream model-load progress" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelInterfaceTests/qwen3OnDemandStreamReportsModelLoadProgress" \
  "model_load_progress"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 stop sequence" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelGenerationTests/qwen3StopsOnConfiguredStopSequence" \
  "stop_sequence"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 rotating and quantized KV cache options" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelGenerationTests/qwen3RunsRuntimeKVCacheOptions" \
  "runtime_kv_cache"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 memory guard admission decisions" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelGenerationTests/qwen3RecordsMemoryGuardAdmissionDecisions" \
  "memory_guard"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 redacted request summary observability" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelGenerationTests/qwen3RecordsRedactedRequestSummaryObservability" \
  "observability"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 greedy and constrained decode paths" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelGenerationTests/qwen3ReportsGreedyAndConstrainedDecodePaths" \
  "greedy_constrained_decode"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 tool call rendering" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3EmitsParseableToolCall" \
  "tool_call_rendering"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 translated tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3ToolStreamEmitsToolEventsWithoutProtocolMarkup" \
  "tool_stream_translation"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 schema-normalized tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3ToolStreamNormalizesArgumentsWithToolSchemas" \
  "tool_schema_normalization"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3ConstrainedNativeToolStreamEmitsTypedArguments" \
  "native_tool_constraints"

run_model_feature_test \
  "gemma-4-e2b-it-4bit" \
  "Gemma 4 native translated tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/gemma4NativeToolStreamEmitsStructuredToolEvents" \
  "native_tool_stream_translation"

run_model_feature_test \
  "gemma-4-e2b-it-4bit" \
  "Gemma 4 constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/gemma4ConstrainedNativeToolStreamEmitsTypedArguments" \
  "native_tool_constraints"

run_model_feature_test \
  "mistral-7b-v0.3-4bit" \
  "Mistral constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/mistralConstrainedNativeToolStreamEmitsTypedArguments" \
  "native_tool_constraints"

run_model_feature_test \
  "glm-4-9b-0414-4bit" \
  "GLM constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/glmConstrainedNativeToolStreamEmitsTypedArguments" \
  "native_tool_constraints"

run_model_feature_test \
  "gpt-oss" \
  "Harmony constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/harmonyConstrainedNativeToolStreamEmitsTypedArguments" \
  "native_tool_constraints"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 JSON schema constraints" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/qwen3GeneratesValidJSONThroughTokenLevelSchemaConstraints" \
  "json_schema_constraints"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 finite choice constraints" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/qwen3GeneratesOnlyOneFiniteChoiceTokenSequence" \
  "finite_choice_constraints"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 persistent prompt cache restore" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelPersistentPromptCacheTests/qwen3RestoresPersistentPromptCacheAcrossSessions" \
  "persistent_prompt_cache"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 continuous-batch prompt cache reuse" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelBatchCacheTests/qwen3ReusesMemoryPromptCacheThroughContinuousBatching" \
  "continuous_batch_prompt_cache"

COUNT=0
for MODEL in "${MODELS[@]}"; do
  COUNT=$((COUNT + 1))
  IFS=$'\t' read -r ID DISPLAY_NAME ARCHITECTURE TAGS <<<"$MODEL"
  log_benchmark_line ""
  log_benchmark_line "[$COUNT/${#MODELS[@]}] $DISPLAY_NAME"
  log_benchmark_line "   id:           $ID"
  log_benchmark_line "   architecture: $ARCHITECTURE"
  log_benchmark_line "   tags:         $TAGS"

  run_swift_test \
    "$ID generation" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelGenerationTests/selectedCatalogModelsLoadAndGenerate" \
    "$ID" \
    "generation"
  if [[ "$LAST_TEST_STATUS" -ne 0 ]]; then
    log_benchmark_line "   skipping follow-up checks for $ID because generation did not pass"
    continue
  fi

  run_swift_test \
    "$ID sampling and logits controls" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelSamplingTests/selectedCatalogModelsApplySamplingAndLogitsControls" \
    "$ID" \
    "sampling_logits"

  run_swift_test \
    "$ID stream lifecycle phase boundaries" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelInterfaceTests/selectedModelsReportStreamLifecyclePhaseBoundaries" \
    "$ID" \
    "stream_lifecycle"

  run_swift_test \
    "$ID on-demand stream model-load progress" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelInterfaceTests/selectedModelsReportOnDemandStreamModelLoadProgress" \
    "$ID" \
    "model_load_progress"

  run_swift_test \
    "$ID memory guard admission decisions" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelGenerationTests/selectedModelsRecordMemoryGuardAdmissionDecisions" \
    "$ID" \
    "memory_guard"

  run_swift_test \
    "$ID redacted request summary observability" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelGenerationTests/selectedModelsRecordRedactedRequestSummaryObservability" \
    "$ID" \
    "observability"

  run_swift_test \
    "$ID greedy and constrained decode paths" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelGenerationTests/selectedModelsReportGreedyAndConstrainedDecodePaths" \
    "$ID" \
    "greedy_constrained_decode"

  if [[ "$ARCHITECTURE" != "mamba" && "$ARCHITECTURE" != "mamba2" && "$ARCHITECTURE" != "rwkv7" ]]; then
    run_swift_test \
      "$ID rotating and quantized KV cache options" \
      "$MODEL_TIMEOUT_SECONDS" \
      "MLXRealModelTests.MLXRealModelGenerationTests/selectedAttentionModelsRunRuntimeKVCacheOptions" \
      "$ID" \
      "runtime_kv_cache"
  fi

  run_swift_test \
    "$ID continuous-batch prompt cache reuse" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelBatchCacheTests/selectedModelsReuseMemoryPromptCacheThroughContinuousBatching" \
    "$ID" \
    "continuous_batch_prompt_cache"

  run_swift_test \
    "$ID persistent prompt cache restore" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelPersistentPromptCacheTests/selectedModelsRestorePersistentPromptCacheAcrossSessions" \
    "$ID" \
    "persistent_prompt_cache"

  if [[ "$TAGS" != *"native-template-only"* ]]; then
    run_swift_test \
      "$ID session-style request" \
      "$MODEL_TIMEOUT_SECONDS" \
      "MLXRealModelTests.MLXRealModelInterfaceTests/selectedModelsRunRenderedSessionStyleRequests" \
      "$ID" \
      "session_style_request"
  fi

  run_swift_test \
    "$ID JSON schema constraints" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/selectedModelsGenerateValidJSONThroughTokenLevelSchemaConstraints" \
    "$ID" \
    "json_schema_constraints"

  run_swift_test \
    "$ID finite choice constraints" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/selectedModelsGenerateOnlyOneFiniteChoiceTokenSequence" \
    "$ID" \
    "finite_choice_constraints"

  run_swift_test \
    "$ID token-level grammar constraint" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/selectedArchitecturesForceGrammarValidFirstToken" \
    "$ID" \
    "token_grammar_constraints"

  if architecture_supports_native_tool_constraints "$ARCHITECTURE"; then
    run_swift_test \
      "$ID constrained native tool streaming" \
      "$MODEL_TIMEOUT_SECONDS" \
      "MLXRealModelTests.MLXRealModelToolCallingTests/selectedNativeToolModelsEmitConstrainedTypedToolCalls" \
      "$ID" \
      "native_tool_constraints,native_tool_stream_translation"
  fi

  if [[ "$TAGS" == *"stress"* ]]; then
    run_swift_test \
      "$ID stress generation" \
      "$MODEL_TIMEOUT_SECONDS" \
      "MLXRealModelTests.MLXRealModelStressTests/selectedModelsSurviveRepeatedGeneration" \
      "$ID" \
      "stress_generation"
  fi
done

log_benchmark_line ""
if [[ "${#FAILURES[@]}" -eq 0 ]]; then
  log_benchmark_line "All real-model test commands passed."
  write_benchmark_summary "passed"
  if coverage_passed && benchmark_coverage_passed; then
    exit 0
  fi
  if ! coverage_passed; then
    log_benchmark_line "Feature coverage validation failed:"
    log_coverage_failures
  fi
  if ! benchmark_coverage_passed; then
    log_benchmark_line "Benchmark coverage validation failed:"
    log_benchmark_coverage_failures
  fi
  exit 1
fi

log_benchmark_line "Real-model validation failures:"
for FAILURE in "${FAILURES[@]}"; do
  log_benchmark_line "- $FAILURE"
done
write_benchmark_summary "failed"
exit 1
