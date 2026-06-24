#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATALOG_PATH="${MLX_MODEL_CATALOG_PATH:-$ROOT_DIR/Tests/MLXRealModelTests/Resources/model-catalog.json}"
MODEL_DIR="${MLX_TEST_MODELS_DIR:-$ROOT_DIR/.models}"
FILTER="${MLX_MODEL_FILTER:-}"
ASSUME_YES="${MLX_ASSUME_YES:-0}"
DOWNLOADER="${MLX_MODEL_DOWNLOADER:-auto}"
HF_MAX_WORKERS="${MLX_HF_MAX_WORKERS:-4}"
ALLOW_OVERSIZED_MODELS="${MLX_ALLOW_OVERSIZED_MODELS:-0}"
DOWNLOAD_TIMEOUT_SECONDS="${MLX_MODEL_DOWNLOAD_TIMEOUT_SECONDS:-1800}"
MODEL_STORAGE_TIMEOUT_SECONDS="${MLX_MODEL_STORAGE_TIMEOUT_SECONDS:-10}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

case "$DOWNLOADER" in
auto)
  if ! command -v hf >/dev/null 2>&1 &&
    { ! command -v git >/dev/null 2>&1 || ! command -v git-lfs >/dev/null 2>&1; }; then
    echo "Either hf or git plus git-lfs is required."
    echo "Install hf with: brew install huggingface-cli"
    echo "Install git-lfs with: brew install git-lfs"
    exit 1
  fi
  ;;
hf)
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf is required. Install with: brew install huggingface-cli"
    exit 1
  fi
  ;;
git | git-lfs)
  if ! command -v git >/dev/null 2>&1 || ! command -v git-lfs >/dev/null 2>&1; then
    echo "git and git-lfs are required. Install git-lfs with: brew install git-lfs"
    exit 1
  fi
  ;;
*)
  echo "Unknown MLX_MODEL_DOWNLOADER '$DOWNLOADER'. Use auto, hf, or git."
  exit 1
  ;;
esac

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

download_with_hf() {
  local repository="$1"
  local target="$2"

  command -v hf >/dev/null 2>&1 || return 127
  mkdir -p "$target"
  run_with_timeout \
    "$DOWNLOAD_TIMEOUT_SECONDS" \
    hf download "$repository" --local-dir "$target" --max-workers "$HF_MAX_WORKERS"
}

download_with_git_lfs() {
  local repository="$1"
  local target="$2"
  local tmp_target="$target.tmp"

  command -v git >/dev/null 2>&1 || return 127
  command -v git-lfs >/dev/null 2>&1 || return 127
  git lfs install >/dev/null

  rm -rf "$tmp_target"
  if run_with_timeout \
    "$DOWNLOAD_TIMEOUT_SECONDS" \
    git clone --depth 1 "https://huggingface.co/$repository" "$tmp_target"; then
    rm -rf "$tmp_target/.git"
    rm -rf "$target"
    mv "$tmp_target" "$target"
    return 0
  fi
  rm -rf "$tmp_target"
  return 1
}

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
    has_tokenizer = "tokenizer.json" in names or "tokenizer.model" in names
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

download_model() {
  local repository="$1"
  local target="$2"

  case "$DOWNLOADER" in
  auto)
    if command -v hf >/dev/null 2>&1; then
      if download_with_hf "$repository" "$target"; then
        return 0
      fi
      echo "hf download failed for $repository; falling back to git-lfs." >&2
    fi
    download_with_git_lfs "$repository" "$target"
    ;;
  hf)
    download_with_hf "$repository" "$target"
    ;;
  git | git-lfs)
    download_with_git_lfs "$repository" "$target"
    ;;
  *)
    echo "Unknown MLX_MODEL_DOWNLOADER '$DOWNLOADER'. Use auto, hf, or git."
    return 2
    ;;
  esac
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

available_disk_gb() {
  local path="$1"
  mkdir -p "$path"
  df -Pk "$path" | awk 'NR == 2 { print int($4 / 1024 / 1024) }'
}

mkdir -p "$MODEL_DIR"
HOST_MEMORY_GB="${MLX_HOST_MEMORY_GB:-$(host_memory_gb)}"
AVAILABLE_DISK_GB="$(available_disk_gb "$MODEL_DIR")"

echo "MLXFoundationModel test model downloader"
echo "Catalog: $CATALOG_PATH"
echo "Target:  $MODEL_DIR"
echo "Downloader: $DOWNLOADER"
echo "Host RAM: ${HOST_MEMORY_GB} GiB"
echo "Free disk: ${AVAILABLE_DISK_GB} GiB"
echo "Download timeout: ${DOWNLOAD_TIMEOUT_SECONDS}s"
echo "Storage timeout: ${MODEL_STORAGE_TIMEOUT_SECONDS}s"
if [[ -n "$FILTER" ]]; then
  echo "Filter:  $FILTER"
fi
echo

mapfile -t MODELS < <(
  python3 - "$CATALOG_PATH" "$FILTER" <<'PY'
import json
import sys

catalog_path = sys.argv[1]
filter_value = sys.argv[2].lower()

with open(catalog_path, "r", encoding="utf-8") as file:
    models = json.load(file)

for model in models:
    repository = model.get("repository")
    if not repository:
        continue
    fields = [
        model["id"],
        model["displayName"],
        model["architecture"],
        repository,
        model["relativePath"],
        ",".join(model.get("tags", [])),
        str(model.get("minimumMemoryGB", "")),
        str(model.get("minimumDiskGB", "")),
    ]
    haystack = " ".join(fields).lower()
    if filter_value and filter_value not in haystack:
        continue
    print("\t".join(fields))
PY
)

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No downloadable models matched the catalog/filter."
  exit 0
fi

echo "Matched ${#MODELS[@]} downloadable model(s)."
if [[ "$ASSUME_YES" != "1" ]]; then
  echo "These downloads can require many gigabytes. Continue? [y/N]"
  read -r RESPONSE
  if [[ "$RESPONSE" != "y" && "$RESPONSE" != "Y" ]]; then
    echo "Download cancelled."
    exit 0
  fi
fi

FAILED=0
COUNT=0
for MODEL in "${MODELS[@]}"; do
  COUNT=$((COUNT + 1))
  IFS=$'\t' read -r ID DISPLAY_NAME ARCHITECTURE REPOSITORY RELATIVE_PATH TAGS \
    MINIMUM_MEMORY_GB MINIMUM_DISK_GB <<<"$MODEL"
  TARGET="$MODEL_DIR/$RELATIVE_PATH"
  AVAILABLE_DISK_GB="$(available_disk_gb "$MODEL_DIR")"

  echo
  echo "[$COUNT/${#MODELS[@]}] $DISPLAY_NAME"
  echo "Architecture: $ARCHITECTURE"
  echo "Repository:   $REPOSITORY"
  echo "Target:       $TARGET"
  if [[ -n "$MINIMUM_MEMORY_GB" ]]; then
    echo "Minimum RAM:  ${MINIMUM_MEMORY_GB} GiB"
  fi
  if [[ -n "$MINIMUM_DISK_GB" ]]; then
    echo "Minimum disk: ${MINIMUM_DISK_GB} GiB"
  fi

  if [[ "$ALLOW_OVERSIZED_MODELS" != "1" ]] &&
    [[ -n "$MINIMUM_MEMORY_GB" ]] &&
    ((MINIMUM_MEMORY_GB > HOST_MEMORY_GB)); then
    echo "Skipping $ID: requires ${MINIMUM_MEMORY_GB} GiB RAM, host has ${HOST_MEMORY_GB} GiB."
    echo "Set MLX_ALLOW_OVERSIZED_MODELS=1 to download anyway."
    continue
  fi
  if [[ "$ALLOW_OVERSIZED_MODELS" != "1" ]] &&
    [[ -n "$MINIMUM_DISK_GB" ]] &&
    ((MINIMUM_DISK_GB > AVAILABLE_DISK_GB)); then
    echo "Skipping $ID: requires ${MINIMUM_DISK_GB} GiB free disk, host has ${AVAILABLE_DISK_GB} GiB."
    echo "Set MLX_ALLOW_OVERSIZED_MODELS=1 to download anyway."
    continue
  fi

  if is_complete_model_download "$TARGET"; then
    echo "Already downloaded."
    continue
  elif [[ "$?" -eq 124 ]]; then
    FAILED=1
    continue
  fi

  if download_model "$REPOSITORY" "$TARGET" && is_complete_model_download "$TARGET"; then
    echo "Downloaded $ID."
  else
    echo "Failed to download $ID."
    FAILED=1
  fi
done

echo
du -sh "$MODEL_DIR" 2>/dev/null || true
exit "$FAILED"
