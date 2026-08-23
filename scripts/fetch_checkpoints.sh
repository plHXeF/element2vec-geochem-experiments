#!/usr/bin/env bash
set -euo pipefail

REPO="plHXeF/element2vec-geochem-experiments"
TAG="v0.1.0-stage3.6"
DEST="stage3_5_element2vec_v2/outputs/checkpoints"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--force]" >&2
  exit 2
fi

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

download_one() {
  local name="$1"
  local expected="$2"
  local target="$DEST/$name"
  local temporary="$target.part"
  local url="https://github.com/$REPO/releases/download/$TAG/$name"

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

  echo "Downloading $name"
  curl --fail --location --retry 3 --output "$temporary" "$url"
  local actual
  actual="$(hash_file "$temporary")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $name" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
  mv "$temporary" "$target"
  echo "Verified $name"
}

download_one "decoders.pt" "50ea31a13e95edc3f154395fcc0751b69a4c56067546e73b2fcf4b1a76b897d4"
download_one "full_model.pt" "1d9210c0db29f243193df23bd434e98d169a1fe112e87c8809b4b3716397bd1e"
download_one "token_projector.pt" "3e5b6bfb487e66d59523fb9c1fccde8c3460bc67b0254a480d327e036ce15ca4"
download_one "transformer_encoder.pt" "40445719c14a2a3223e3ddfd5e03b9ea5d0d2005a71a1e84ad9ab7e5db545173"

echo "All Element2Vec V2 checkpoints are ready in $DEST"
