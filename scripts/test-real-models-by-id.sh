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

mapfile -t MODELS < <(
  python3 - "$CATALOG_PATH" "$SCOPE" "$MODEL_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

catalog_path = Path(sys.argv[1])
scope = sys.argv[2]
model_dir = Path(sys.argv[3])
selected_ids = {
    value.strip()
    for value in os.environ.get("MLX_REAL_MODEL_IDS", "").split(",")
    if value.strip()
}

with catalog_path.open("r", encoding="utf-8") as file:
    models = json.load(file)


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
echo "Selected model count:  ${#MODELS[@]}"
echo "Per-model timeout:     ${MODEL_TIMEOUT_SECONDS}s"
echo "Feature timeout:       ${FEATURE_TIMEOUT_SECONDS}s"
echo "Generation token cap:  ${GENERATION_TOKENS}"
echo

SWIFT_TEST_FLAGS=(--configuration "$CONFIGURATION" --no-parallel)
COMMON_ENV=(
  MLX_RUN_REAL_MODEL_TESTS=1
  MLX_REAL_MODEL_SCOPE="$SCOPE"
  MLX_REAL_MODEL_GENERATION_TOKENS="$GENERATION_TOKENS"
  MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS="$GENERATION_TIMEOUT_SECONDS"
)
FAILURES=()
LAST_TEST_STATUS=0

run_swift_test() {
  local label="$1"
  local timeout_seconds="$2"
  local filter="$3"
  local model_id="${4:-}"
  local environment=("${COMMON_ENV[@]}")
  if [[ -n "$model_id" ]]; then
    environment+=(MLX_REAL_MODEL_IDS="$model_id")
  fi

  echo "-> $label"
  if run_with_timeout "$timeout_seconds" \
    env "${environment[@]}" swift test "${SWIFT_TEST_FLAGS[@]}" --filter "$filter"; then
    echo "   passed"
    LAST_TEST_STATUS=0
    return 0
  else
    local status=$?
    echo "   failed with exit status $status"
    FAILURES+=("$label (exit status $status)")
    LAST_TEST_STATUS="$status"
    return 0
  fi
}

run_swift_test \
  "catalog metadata" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelCatalogTests"

run_swift_test \
  "Qwen3 sampling and logits controls" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelSamplingTests/qwen3GenerationAppliesSamplingAndLogitsControls"

run_swift_test \
  "Qwen3 rendered text streaming" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelInterfaceTests/qwen3StreamsMultipleChunksFromRenderedTextRequest"

run_swift_test \
  "Qwen3 stop sequence" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelGenerationTests/qwen3StopsOnConfiguredStopSequence"

run_swift_test \
  "Qwen3 tool call rendering" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelToolCallingTests/qwen3EmitsParseableToolCall"

run_swift_test \
  "Qwen3 JSON schema constraints" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/qwen3GeneratesValidJSONThroughTokenLevelSchemaConstraints"

run_swift_test \
  "Qwen3 finite choice constraints" \
  "$FEATURE_TIMEOUT_SECONDS" \
  "MLXRealModelTests.MLXRealModelConstrainedDecodingTests/qwen3GeneratesOnlyOneFiniteChoiceTokenSequence"

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
