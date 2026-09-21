#!/usr/bin/env bash
set -euo pipefail

REPO="plHXeF/element2vec-geochem-experiments"
TAG="v0.1.0-stage3.6"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      [[ $# -ge 2 ]] || { echo "--run-dir requires a path" >&2; exit 2; }
      export GEOCHEM_RUN_DIR="$2"
      shift 2
      ;;
    --force) FORCE=1; shift ;;
    --help|-h) echo "Usage: $0 --run-dir REGISTERED_RUN [--force]"; exit 0 ;;
    *) echo "Usage: $0 --run-dir REGISTERED_RUN [--force]" >&2; exit 2 ;;
  esac
done
if [[ -z "${GEOCHEM_RUN_DIR:-}" ]]; then
  echo "Specify --run-dir for a new registered run before downloading checkpoints." >&2
  exit 2
fi
DEST="$(python -c 'from geochem_workspace import require_run_dir; print(require_run_dir() / "checkpoints")')"
if [[ ! -f "docs/checkpoint_manifest.json" ]]; then
  echo "Run this script from the repository root." >&2
  exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
  hash_file() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  hash_file() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "Neither sha256sum nor shasum is available." >&2
  exit 2
fi

mkdir -p "$DEST"

download_asset() {
  local name="$1"
  local target="$2"
  local temporary="$target.download"
  local url="https://github.com/$REPO/releases/download/$TAG/$name"

  if [[ -f "$target" ]]; then
    echo "Using existing part $name"
    return
  fi
  echo "Downloading $name"
  curl --fail --location --retry 5 --retry-all-errors --output "$temporary" "$url"
  mv "$temporary" "$target"
}

download_one() {
  local name="$1"
  local expected="$2"
  shift 2
  local target="$DEST/$name"
  local temporary="$target.assembling"
  local part_dir="$DEST/.${name}.parts"

  if [[ -f "$target" ]]; then
    local existing
    existing="$(hash_file "$target")"
    if [[ "$existing" == "$expected" ]]; then
      echo "Verified existing $name"
      return
    fi
    if [[ "$FORCE" -ne 1 ]]; then
      echo "$name exists but has the wrong checksum; rerun with --force." >&2
      exit 1
    fi
  fi

  mkdir -p "$part_dir"
  local asset
  for asset in "$@"; do
    download_asset "$asset" "$part_dir/$asset"
  done

  : > "$temporary"
  for asset in "$@"; do
    cat "$part_dir/$asset" >> "$temporary"
  done
  local actual
  actual="$(hash_file "$temporary")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $name" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
  mv "$temporary" "$target"
  rm -rf "$part_dir"
  echo "Verified $name"
}

download_one "decoders.pt" "50ea31a13e95edc3f154395fcc0751b69a4c56067546e73b2fcf4b1a76b897d4" \
  "decoders.pt"
download_one "full_model.pt" "1d9210c0db29f243193df23bd434e98d169a1fe112e87c8809b4b3716397bd1e" \
  "full_model.pt.part-000" "full_model.pt.part-001" "full_model.pt.part-002" \
  "full_model.pt.part-003" "full_model.pt.part-004" "full_model.pt.part-005" \
  "full_model.pt.part-006" "full_model.pt.part-007" "full_model.pt.part-008" \
  "full_model.pt.part-009" "full_model.pt.part-010" "full_model.pt.part-011" \
  "full_model.pt.part-012" "full_model.pt.part-013" "full_model.pt.part-014" \
  "full_model.pt.part-015"
download_one "token_projector.pt" "3e5b6bfb487e66d59523fb9c1fccde8c3460bc67b0254a480d327e036ce15ca4" \
  "token_projector.pt"
download_one "transformer_encoder.pt" "40445719c14a2a3223e3ddfd5e03b9ea5d0d2005a71a1e84ad9ab7e5db545173" \
  "transformer_encoder.pt.part-000" "transformer_encoder.pt.part-001" \
  "transformer_encoder.pt.part-002" "transformer_encoder.pt.part-003" \
  "transformer_encoder.pt.part-004" "transformer_encoder.pt.part-005"

echo "All Element2Vec V2 checkpoints are ready in $DEST"
