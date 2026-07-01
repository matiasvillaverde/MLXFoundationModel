#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATALOG_PATH="${MLX_MODEL_CATALOG_PATH:-$ROOT_DIR/Tests/MLXRealModelTests/Resources/model-catalog.json}"
MODEL_DIR="${MLX_TEST_MODELS_DIR:-$ROOT_DIR/.models}"
MODEL_ID="${MLX_DEMO_MODEL_ID:-qwen3-0.6b-4bit}"
EXAMPLE_ID="${MLX_DEMO_EXAMPLE:-streaming-chat}"
CONFIGURATION="${CONFIGURATION:-release}"
MODEL_STORAGE_TIMEOUT_SECONDS="${MLX_MODEL_STORAGE_TIMEOUT_SECONDS:-10}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

mapfile -t MODEL < <(
  python3 - "$CATALOG_PATH" "$MODEL_ID" <<'PY'
import json
import sys

catalog_path = sys.argv[1]
model_id = sys.argv[2]

with open(catalog_path, "r", encoding="utf-8") as file:
    models = json.load(file)

for model in models:
    if model["id"] == model_id:
        print(model["relativePath"])
        print(model["displayName"])
        sys.exit(0)

print(f"Unknown model id: {model_id}", file=sys.stderr)
sys.exit(1)
PY
)

if [[ "${#MODEL[@]}" -lt 2 ]]; then
  echo "Unknown demo model: $MODEL_ID"
  exit 1
fi

RELATIVE_PATH="${MODEL[0]}"
DISPLAY_NAME="${MODEL[1]}"
MODEL_PATH="$MODEL_DIR/$RELATIVE_PATH"

is_complete_model_download() {
  local target="$1"

  python3 - "$target" "$MODEL_STORAGE_TIMEOUT_SECONDS" <<'PY'
import os
import signal
import sys

target = sys.argv[1]
timeout_seconds = int(sys.argv[2])


def timeout_handler(_signum, _frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(timeout_seconds)

try:
    if not os.path.isdir(target):
        sys.exit(1)

    names = set(os.listdir(target))
    has_config = "config.json" in names
    has_tokenizer = (
        "tokenizer.json" in names
        or "tokenizer.model" in names
        or "cl100k_base.tiktoken" in names
        or "tiktoken.model" in names
        or "hy.tiktoken" in names
    )
    has_weights = (
        "model.safetensors.index.json" in names
        or any(name.endswith(".safetensors") for name in names)
    )
    sys.exit(0 if has_config and has_tokenizer and has_weights else 1)
except TimeoutError:
    print(
        f"Model path did not respond within {timeout_seconds}s: {target}",
        file=sys.stderr,
    )
    print(
        "Check the model volume or set MLX_TEST_MODELS_DIR to a responsive model root.",
        file=sys.stderr,
    )
    sys.exit(124)
except OSError as error:
    print(f"Cannot inspect model path {target}: {error}", file=sys.stderr)
    sys.exit(1)
finally:
    signal.alarm(0)
PY
}

echo "Model: $DISPLAY_NAME"
echo "Path:  $MODEL_PATH"

if is_complete_model_download "$MODEL_PATH"; then
  :
else
  STATUS=$?
  if [[ "$STATUS" -eq 124 ]]; then
    exit "$STATUS"
  fi
  echo "Downloading $DISPLAY_NAME into $MODEL_DIR"
  MLX_ASSUME_YES="${MLX_ASSUME_YES:-1}" \
    MLX_MODEL_FILTER="$MODEL_ID" \
    MLX_TEST_MODELS_DIR="$MODEL_DIR" \
    bash "$ROOT_DIR/scripts/download-test-models.sh"
fi

if is_complete_model_download "$MODEL_PATH"; then
  :
else
  STATUS=$?
  if [[ "$STATUS" -eq 124 ]]; then
    exit "$STATUS"
  fi
  echo "Model download is incomplete: $MODEL_PATH"
  exit 1
fi

echo "Running $DISPLAY_NAME with example '$EXAMPLE_ID'"
swift run --configuration "$CONFIGURATION" FoundationModelsPlayground \
  --model-path "$MODEL_PATH" \
  --model-id "$MODEL_ID" \
  --example "$EXAMPLE_ID" \
  "$@"
