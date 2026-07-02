#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MLX_PROFILE_MODEL_PATH:-$ROOT_DIR/.models/Qwen3-0.6B-4bit}"
MODEL_ID="${MLX_PROFILE_MODEL_ID:-qwen3-0.6b-4bit}"
EXAMPLE_ID="${MLX_PROFILE_EXAMPLE:-streaming-chat}"
PROMPT="${MLX_PROFILE_PROMPT:-}"
INSTRUCTIONS="${MLX_PROFILE_INSTRUCTIONS:-}"
MAX_TOKENS="${MLX_PROFILE_MAX_TOKENS:-}"
TEMPLATE="${MLX_PROFILE_TEMPLATE:-Time Profiler}"
TIME_LIMIT="${MLX_PROFILE_TIME_LIMIT:-20s}"
OUTPUT_DIR="${MLX_PROFILE_OUTPUT_DIR:-$ROOT_DIR/.build/reports/profiles}"
MIN_FREE_GIB="${MLX_PROFILE_MIN_FREE_GIB:-8}"
PRODUCT_NAME="${MLX_PROFILE_PRODUCT:-FoundationModelsPlayground}"
BUILD_CONFIGURATION="${CONFIGURATION:-release}"
MODEL_STORAGE_TIMEOUT_SECONDS="${MLX_MODEL_STORAGE_TIMEOUT_SECONDS:-10}"
RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

usage() {
  cat <<'USAGE'
Profile a real MLX-backed Foundation Models playground run with Instruments.

Environment:
  MLX_PROFILE_MODEL_PATH       Model directory. Defaults to .models/Qwen3-0.6B-4bit.
  MLX_PROFILE_MODEL_ID         Catalog/model identifier. Defaults to qwen3-0.6b-4bit.
  MLX_PROFILE_EXAMPLE          Playground example id. Defaults to streaming-chat.
  MLX_PROFILE_PROMPT           Optional benchmark prompt. Overrides MLX_PROFILE_EXAMPLE.
  MLX_PROFILE_INSTRUCTIONS     Optional system instructions for MLX_PROFILE_PROMPT.
  MLX_PROFILE_MAX_TOKENS       Optional generated-token limit for MLX_PROFILE_PROMPT.
  MLX_PROFILE_TEMPLATE         xctrace template. Defaults to Time Profiler.
  MLX_PROFILE_TIME_LIMIT       xctrace time limit. Defaults to 20s.
  MLX_PROFILE_OUTPUT_DIR       Output directory. Defaults to .build/reports/profiles.
  MLX_PROFILE_MIN_FREE_GIB     Refuse to start below this free-space floor. Defaults to 8.
  MLX_MODEL_STORAGE_TIMEOUT_SECONDS
                                Refuse to start if the model path does not respond within this many seconds.
  CONFIGURATION                Swift build configuration. Defaults to release.

Examples:
  scripts/profile-real-model.sh
  MLX_PROFILE_TEMPLATE='Metal System Trace' scripts/profile-real-model.sh
  MLX_PROFILE_TEMPLATE='File Activity' MLX_PROFILE_TIME_LIMIT=15s scripts/profile-real-model.sh
USAGE
}

write_run_metadata() {
  local xctrace_status="$1"
  local toc_exported="$2"
  MLX_PROFILE_RUN_JSON_PATH="$RUN_JSON_PATH" \
  MLX_PROFILE_ROOT_DIR="$ROOT_DIR" \
  MLX_PROFILE_RUN_STARTED_UTC="$RUN_STARTED_UTC" \
  MLX_PROFILE_RUN_ID="$RUN_ID" \
  MLX_PROFILE_MODEL_ID_VALUE="$MODEL_ID" \
  MLX_PROFILE_MODEL_PATH_VALUE="$MODEL_PATH" \
  MLX_PROFILE_EXAMPLE_VALUE="$EXAMPLE_ID" \
  MLX_PROFILE_PROMPT_OVERRIDE="$([[ -n "$PROMPT" ]] && echo "true" || echo "false")" \
  MLX_PROFILE_INSTRUCTIONS_OVERRIDE="$([[ -n "$INSTRUCTIONS" ]] && echo "true" || echo "false")" \
  MLX_PROFILE_MAX_TOKENS_VALUE="${MAX_TOKENS:-}" \
  MLX_PROFILE_TEMPLATE_VALUE="$TEMPLATE" \
  MLX_PROFILE_TIME_LIMIT_VALUE="$TIME_LIMIT" \
  MLX_PROFILE_PRODUCT_VALUE="$PRODUCT_NAME" \
  MLX_PROFILE_BUILD_CONFIGURATION="$BUILD_CONFIGURATION" \
  MLX_PROFILE_TRACE_PATH="$TRACE_PATH" \
  MLX_PROFILE_STDOUT_PATH="$STDOUT_PATH" \
  MLX_PROFILE_TOC_PATH="$TOC_PATH" \
  MLX_PROFILE_TOC_EXPORTED="$toc_exported" \
  MLX_PROFILE_XCTRACE_STATUS="$xctrace_status" \
  python3 <<'PY'
import hashlib
import json
import os
import platform
import subprocess


def command_output(command):
    try:
        return subprocess.check_output(
            command,
            cwd=os.environ["MLX_PROFILE_ROOT_DIR"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty():
    unstaged = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=os.environ["MLX_PROFILE_ROOT_DIR"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=os.environ["MLX_PROFILE_ROOT_DIR"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    return unstaged != 0 or staged != 0


def file_size(path):
    if not path or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except OSError:
                pass
    return total


def host_memory_bytes():
    if platform.system() == "Darwin":
        value = command_output(["sysctl", "-n", "hw.memsize"])
        if value and value.isdigit():
            return int(value)
    return None


model_path = os.environ["MLX_PROFILE_MODEL_PATH_VALUE"]
model_path_real = os.path.realpath(model_path)
trace_path = os.environ["MLX_PROFILE_TRACE_PATH"]
stdout_path = os.environ["MLX_PROFILE_STDOUT_PATH"]
toc_path = os.environ["MLX_PROFILE_TOC_PATH"]

metadata = {
    "schema_version": 1,
    "run_id": os.environ["MLX_PROFILE_RUN_ID"],
    "started_utc": os.environ["MLX_PROFILE_RUN_STARTED_UTC"],
    "model": {
        "id": os.environ["MLX_PROFILE_MODEL_ID_VALUE"],
        "path_basename": os.path.basename(model_path_real),
        "path_fingerprint": hashlib.sha256(model_path_real.encode("utf-8")).hexdigest(),
    },
    "workload": {
        "example_id": os.environ["MLX_PROFILE_EXAMPLE_VALUE"],
        "prompt_override": os.environ["MLX_PROFILE_PROMPT_OVERRIDE"] == "true",
        "instructions_override": os.environ["MLX_PROFILE_INSTRUCTIONS_OVERRIDE"] == "true",
        "max_tokens": os.environ["MLX_PROFILE_MAX_TOKENS_VALUE"] or None,
    },
    "profile": {
        "template": os.environ["MLX_PROFILE_TEMPLATE_VALUE"],
        "time_limit": os.environ["MLX_PROFILE_TIME_LIMIT_VALUE"],
        "product": os.environ["MLX_PROFILE_PRODUCT_VALUE"],
        "configuration": os.environ["MLX_PROFILE_BUILD_CONFIGURATION"],
        "xctrace_status": int(os.environ["MLX_PROFILE_XCTRACE_STATUS"]),
        "toc_exported": os.environ["MLX_PROFILE_TOC_EXPORTED"] == "true",
    },
    "artifacts": {
        "trace": trace_path,
        "trace_bytes": file_size(trace_path),
        "stdout": stdout_path,
        "stdout_bytes": file_size(stdout_path),
        "toc": toc_path if os.path.exists(toc_path) else None,
        "toc_bytes": file_size(toc_path),
    },
    "host": {
        "os": platform.platform(),
        "macos_version": command_output(["sw_vers", "-productVersion"]),
        "xcode_version": command_output(["xcodebuild", "-version"]),
        "swift_version": command_output(["swift", "--version"]),
        "memory_bytes": host_memory_bytes(),
    },
    "source": {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
    },
}

run_json_path = os.environ["MLX_PROFILE_RUN_JSON_PATH"]
os.makedirs(os.path.dirname(run_json_path), exist_ok=True)
with open(run_json_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2, sort_keys=True)
    file.write("\n")
PY
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "swift is required" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

if ! TEMPLATES="$(xcrun xctrace list templates)"; then
  echo "xcrun xctrace is required" >&2
  exit 1
fi

python3 - "$MODEL_PATH" "$MODEL_STORAGE_TIMEOUT_SECONDS" <<'PY'
import os
import signal
import sys

model_path = sys.argv[1]
timeout_seconds = int(sys.argv[2])


def timeout_handler(_signum, _frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(timeout_seconds)

try:
    if not os.path.isdir(model_path):
        print(f"Model path does not exist: {model_path}", file=sys.stderr)
        print(
            "Download smoke models with: MLX_ASSUME_YES=1 MLX_MODEL_FILTER=smoke make download-test-models",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "rb") as file:
        file.read(1)

    has_tokenizer = any(
        os.path.isfile(os.path.join(model_path, filename))
        for filename in ("tokenizer.json", "tokenizer.model", "cl100k_base.tiktoken", "tiktoken.model", "hy.tiktoken")
    )
    has_weights = False
    with os.scandir(model_path) as entries:
        for entry in entries:
            if entry.is_file() and (
                entry.name == "model.safetensors.index.json"
                or entry.name.endswith(".safetensors")
            ):
                has_weights = True
                break

    if not has_tokenizer:
        print(f"Model path is missing tokenizer files: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not has_weights:
        print(f"Model path is missing safetensors weights: {model_path}", file=sys.stderr)
        sys.exit(1)
except FileNotFoundError as error:
    print(f"Model path is incomplete: {error.filename}", file=sys.stderr)
    sys.exit(1)
except TimeoutError:
    print(
        f"Model path did not respond within {timeout_seconds}s: {model_path}",
        file=sys.stderr,
    )
    print(
        "Check the model volume or set MLX_PROFILE_MODEL_PATH to a responsive model directory.",
        file=sys.stderr,
    )
    sys.exit(124)
except OSError as error:
    print(f"Cannot inspect model path {model_path}: {error}", file=sys.stderr)
    sys.exit(1)
finally:
    signal.alarm(0)
PY

if ! grep -Fxq "$TEMPLATE" <<<"$TEMPLATES"; then
  echo "xctrace template is not installed: $TEMPLATE" >&2
  echo "Installed templates:" >&2
  printf '%s\n' "$TEMPLATES" >&2
  exit 1
fi

free_gib() {
  df -g "$ROOT_DIR" | awk 'NR == 2 { print $4 }'
}

AVAILABLE_GIB="$(free_gib)"
if [[ "$AVAILABLE_GIB" =~ ^[0-9]+$ ]] && ((AVAILABLE_GIB < MIN_FREE_GIB)); then
  echo "Refusing to profile with only ${AVAILABLE_GIB} GiB free; need ${MIN_FREE_GIB} GiB." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
SAFE_TEMPLATE="$(printf '%s' "$TEMPLATE" | tr '[:upper:] ' '[:lower:]-' | tr -cd '[:alnum:]-')"
RUN_LABEL="$EXAMPLE_ID"
if [[ -n "$PROMPT" ]]; then
  RUN_LABEL="benchmark"
fi
TRACE_PATH="$OUTPUT_DIR/${MODEL_ID}-${RUN_LABEL}-${SAFE_TEMPLATE}-${RUN_ID}.trace"
STDOUT_PATH="$OUTPUT_DIR/${MODEL_ID}-${RUN_LABEL}-${SAFE_TEMPLATE}-${RUN_ID}.stdout"
TOC_PATH="$OUTPUT_DIR/${MODEL_ID}-${RUN_LABEL}-${SAFE_TEMPLATE}-${RUN_ID}-toc.xml"
RUN_JSON_PATH="$OUTPUT_DIR/${MODEL_ID}-${RUN_LABEL}-${SAFE_TEMPLATE}-${RUN_ID}-run.json"

echo "Building $PRODUCT_NAME ($BUILD_CONFIGURATION)..."
swift build --configuration "$BUILD_CONFIGURATION" --product "$PRODUCT_NAME"

BINARY_PATH="$ROOT_DIR/.build/$BUILD_CONFIGURATION/$PRODUCT_NAME"
if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Built product is not executable: $BINARY_PATH" >&2
  exit 1
fi

echo "Recording $TEMPLATE for $TIME_LIMIT"
echo "Trace:  $TRACE_PATH"
echo "Stdout: $STDOUT_PATH"
echo "Run metadata: $RUN_JSON_PATH"

PLAYGROUND_ARGS=(
  --model-path "$MODEL_PATH"
  --model-id "$MODEL_ID"
  --example "$EXAMPLE_ID"
)
if [[ -n "$PROMPT" ]]; then
  PLAYGROUND_ARGS+=(--prompt "$PROMPT")
fi
if [[ -n "$INSTRUCTIONS" ]]; then
  PLAYGROUND_ARGS+=(--instructions "$INSTRUCTIONS")
fi
if [[ -n "$MAX_TOKENS" ]]; then
  PLAYGROUND_ARGS+=(--max-tokens "$MAX_TOKENS")
fi

set +e
xcrun xctrace record \
  --template "$TEMPLATE" \
  --time-limit "$TIME_LIMIT" \
  --output "$TRACE_PATH" \
  --target-stdout "$STDOUT_PATH" \
  --no-prompt \
  --launch -- "$BINARY_PATH" \
  "${PLAYGROUND_ARGS[@]}"
XCTRACE_STATUS="$?"
set -e

if [[ "$XCTRACE_STATUS" -ne 0 ]]; then
  if [[ -d "$TRACE_PATH" ]]; then
    echo "xctrace exited with status $XCTRACE_STATUS after saving the trace; continuing."
  else
    echo "xctrace failed with status $XCTRACE_STATUS before writing a trace." >&2
    exit "$XCTRACE_STATUS"
  fi
fi

TOC_EXPORTED="false"
if xcrun xctrace export --input "$TRACE_PATH" --toc --output "$TOC_PATH" >/dev/null 2>&1; then
  TOC_EXPORTED="true"
  echo "Trace TOC: $TOC_PATH"
else
  rm -f "$TOC_PATH"
  echo "Trace TOC export skipped; open the trace in Instruments for details."
fi

write_run_metadata "$XCTRACE_STATUS" "$TOC_EXPORTED"
echo "Run metadata: $RUN_JSON_PATH"

echo "Generated output:"
if [[ -s "$STDOUT_PATH" ]]; then
  sed -n '1,120p' "$STDOUT_PATH"
else
  echo "(target stdout was empty)"
fi

echo
du -sh "$TRACE_PATH" "$STDOUT_PATH" "$RUN_JSON_PATH" 2>/dev/null || true
