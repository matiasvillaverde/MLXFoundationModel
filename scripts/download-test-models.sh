#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATALOG_PATH="${MLX_MODEL_CATALOG_PATH:-$ROOT_DIR/Tests/MLXRealModelTests/Resources/model-catalog.json}"
MODEL_DIR="${MLX_TEST_MODELS_DIR:-$ROOT_DIR/.models}"
FILTER="${MLX_MODEL_FILTER:-}"
ASSUME_YES="${MLX_ASSUME_YES:-0}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required. Install with: brew install git-lfs"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

mkdir -p "$MODEL_DIR"
git lfs install >/dev/null

echo "MLXFoundationModel test model downloader"
echo "Catalog: $CATALOG_PATH"
echo "Target:  $MODEL_DIR"
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
  IFS=$'\t' read -r ID DISPLAY_NAME ARCHITECTURE REPOSITORY RELATIVE_PATH TAGS <<<"$MODEL"
  TARGET="$MODEL_DIR/$RELATIVE_PATH"

  echo
  echo "[$COUNT/${#MODELS[@]}] $DISPLAY_NAME"
  echo "Architecture: $ARCHITECTURE"
  echo "Repository:   $REPOSITORY"
  echo "Target:       $TARGET"

  if [[ -f "$TARGET/config.json" && -f "$TARGET/tokenizer.json" ]] &&
    find "$TARGET" -maxdepth 1 \( -name '*.safetensors' -o -name 'model.safetensors.index.json' \) \
      -type f -print -quit | grep -q .; then
    echo "Already downloaded."
    continue
  fi

  TMP_TARGET="$TARGET.tmp"
  rm -rf "$TMP_TARGET"
  if git clone --depth 1 "https://huggingface.co/$REPOSITORY" "$TMP_TARGET"; then
    rm -rf "$TARGET"
    mv "$TMP_TARGET" "$TARGET"
    echo "Downloaded $ID."
  else
    rm -rf "$TMP_TARGET"
    echo "Failed to download $ID."
    FAILED=1
  fi
done

echo
du -sh "$MODEL_DIR" 2>/dev/null || true
exit "$FAILED"
