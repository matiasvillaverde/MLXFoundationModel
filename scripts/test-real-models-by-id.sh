#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATALOG_PATH="${MLX_MODEL_CATALOG_PATH:-$ROOT_DIR/Tests/MLXRealModelTests/Resources/model-catalog.json}"
MODEL_DIR="${MLX_TEST_MODELS_DIR:-$ROOT_DIR/.models}"
SCOPE="${MLX_REAL_MODEL_SCOPE:-relevant}"
CONFIGURATION="${CONFIGURATION:-debug}"
MODEL_TIMEOUT_SECONDS="${MLX_REAL_MODEL_TIMEOUT_SECONDS:-1200}"
FEATURE_TIMEOUT_SECONDS="${MLX_REAL_MODEL_FEATURE_TIMEOUT_SECONDS:-900}"
GENERATION_TOKENS="${MLX_REAL_MODEL_GENERATION_TOKENS:-2}"
GENERATION_TIMEOUT_SECONDS="${MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS:-120}"
ALLOW_OVERSIZED_MODELS="${MLX_ALLOW_OVERSIZED_MODELS:-0}"
SKIP_BUILD_AFTER_FIRST="${MLX_REAL_MODEL_SKIP_BUILD_AFTER_FIRST:-1}"
DRY_RUN="${MLX_REAL_MODEL_DRY_RUN:-0}"

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
    has_tokenizer = (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()
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
    if minimum_memory_gb is not None and minimum_memory_gb > host_memory_gb:
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
echo "Generation token cap:  ${GENERATION_TOKENS}"
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

SWIFT_TEST_FLAGS=(--configuration "$CONFIGURATION" --no-parallel)
COMMON_ENV=(
  MLX_RUN_REAL_MODEL_TESTS=1
  MLX_REAL_MODEL_SCOPE="$SCOPE"
  MLX_REAL_MODEL_GENERATION_TOKENS="$GENERATION_TOKENS"
  MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS="$GENERATION_TIMEOUT_SECONDS"
  MLX_HOST_MEMORY_GB="$HOST_MEMORY_GB"
  MLX_ALLOW_OVERSIZED_MODELS="$ALLOW_OVERSIZED_MODELS"
)
FAILURES=()
LAST_TEST_STATUS=0
SWIFT_TEST_INVOCATION_COUNT=0

run_swift_test() {
  local label="$1"
  local timeout_seconds="$2"
  local filter="$3"
  local model_id="${4:-}"
  local environment=("${COMMON_ENV[@]}")
  local swift_test_flags=("${SWIFT_TEST_FLAGS[@]}")
  if [[ -n "$model_id" ]]; then
    environment+=(MLX_REAL_MODEL_IDS="$model_id")
  fi
  if [[ "$SKIP_BUILD_AFTER_FIRST" == "1" && "$SWIFT_TEST_INVOCATION_COUNT" -gt 0 ]]; then
    swift_test_flags+=(--skip-build)
  fi

  echo "-> $label"
  if run_with_timeout "$timeout_seconds" \
    env "${environment[@]}" swift test "${swift_test_flags[@]}" --filter "$filter"; then
    echo "   passed"
    LAST_TEST_STATUS=0
    SWIFT_TEST_INVOCATION_COUNT=$((SWIFT_TEST_INVOCATION_COUNT + 1))
    return 0
  else
    local status=$?
    echo "   failed with exit status $status"
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

  if model_is_selected "$model_id"; then
    run_swift_test "$label" "$timeout_seconds" "$filter" "$model_id"
  else
    echo "-> $label"
    echo "   skipped: $model_id is not selected for scope '$SCOPE'"
  fi
}

run_swift_test \
  "catalog metadata" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelCatalogTests"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 sampling and logits controls" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelSamplingTests/qwen3GenerationAppliesSamplingAndLogitsControls"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 rendered text streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelInterfaceTests/qwen3StreamsMultipleChunksFromRenderedTextRequest"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 stop sequence" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelGenerationTests/qwen3StopsOnConfiguredStopSequence"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 tool call rendering" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3EmitsParseableToolCall"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 translated tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3ToolStreamEmitsToolEventsWithoutProtocolMarkup"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 schema-normalized tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3ToolStreamNormalizesArgumentsWithToolSchemas"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3ConstrainedNativeToolStreamEmitsTypedArguments"

run_model_feature_test \
  "gemma-4-e2b-it-4bit" \
  "Gemma 4 native translated tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/gemma4NativeToolStreamEmitsStructuredToolEvents"

run_model_feature_test \
  "gemma-4-e2b-it-4bit" \
  "Gemma 4 constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/gemma4ConstrainedNativeToolStreamEmitsTypedArguments"

run_model_feature_test \
  "mistral-7b-v0.3-4bit" \
  "Mistral constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/mistralConstrainedNativeToolStreamEmitsTypedArguments"

run_model_feature_test \
  "glm-4-9b-0414-4bit" \
  "GLM constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/glmConstrainedNativeToolStreamEmitsTypedArguments"

run_model_feature_test \
  "gpt-oss" \
  "Harmony constrained native tool streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/harmonyConstrainedNativeToolStreamEmitsTypedArguments"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 JSON schema constraints" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/qwen3GeneratesValidJSONThroughTokenLevelSchemaConstraints"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 finite choice constraints" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/qwen3GeneratesOnlyOneFiniteChoiceTokenSequence"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 persistent prompt cache restore" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelPersistentPromptCacheTests/qwen3RestoresPersistentPromptCacheAcrossSessions"

run_model_feature_test \
  "qwen3-0.6b-4bit" \
  "Qwen3 continuous-batch prompt cache reuse" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelBatchCacheTests/qwen3ReusesMemoryPromptCacheThroughContinuousBatching"

COUNT=0
for MODEL in "${MODELS[@]}"; do
  COUNT=$((COUNT + 1))
  IFS=$'\t' read -r ID DISPLAY_NAME ARCHITECTURE TAGS <<<"$MODEL"
  echo
  echo "[$COUNT/${#MODELS[@]}] $DISPLAY_NAME"
  echo "   id:           $ID"
  echo "   architecture: $ARCHITECTURE"
  echo "   tags:         $TAGS"

  run_swift_test \
    "$ID generation" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelGenerationTests/selectedCatalogModelsLoadAndGenerate" \
    "$ID"
  if [[ "$LAST_TEST_STATUS" -ne 0 ]]; then
    echo "   skipping follow-up checks for $ID because generation did not pass"
    continue
  fi

  if [[ "$TAGS" != *"native-template-only"* ]]; then
    run_swift_test \
      "$ID session-style request" \
      "$MODEL_TIMEOUT_SECONDS" \
      "MLXRealModelTests.MLXRealModelInterfaceTests/selectedModelsRunRenderedSessionStyleRequests" \
      "$ID"
  fi

  run_swift_test \
    "$ID token-level grammar constraint" \
    "$MODEL_TIMEOUT_SECONDS" \
    "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/selectedArchitecturesForceGrammarValidFirstToken" \
    "$ID"
done

echo
if [[ "${#FAILURES[@]}" -eq 0 ]]; then
  echo "All per-model real validations passed."
  exit 0
fi

echo "Real-model validation failures:"
for FAILURE in "${FAILURES[@]}"; do
  echo "- $FAILURE"
done
exit 1
