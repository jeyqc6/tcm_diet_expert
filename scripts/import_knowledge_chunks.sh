#!/usr/bin/env bash
# Import pre-built knowledge_chunks vectors into the Docker Postgres from a Release asset.
# Usage (from repo root):
#   ./scripts/import_knowledge_chunks.sh
#   ./scripts/import_knowledge_chunks.sh /path/to/knowledge_chunks.sql.gz
#   KB_RELEASE_TAG=v0.1.0-kb ./scripts/import_knowledge_chunks.sh
set -euo pipefail

REPO="${GITHUB_REPO:-jeyqc6/tcm_diet_expert}"
TAG="${KB_RELEASE_TAG:-v0.1.0-kb}"
ASSET_NAME="knowledge_chunks.sql.gz"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_CACHE="${ROOT}/data/${ASSET_NAME}"
ARCHIVE="${1:-$DEFAULT_CACHE}"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET_NAME}"

cd "$ROOT"

if ! docker compose ps --status running postgres 2>/dev/null | grep -q postgres; then
  echo "Postgres is not running. Start it first: docker compose up -d postgres" >&2
  exit 1
fi

if [[ ! -f "$ARCHIVE" ]]; then
  mkdir -p "$(dirname "$ARCHIVE")"
  echo "Downloading ${DOWNLOAD_URL} ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$ARCHIVE" "$DOWNLOAD_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$ARCHIVE" "$DOWNLOAD_URL"
  else
    echo "Need curl or wget to download the snapshot." >&2
    exit 1
  fi
fi

echo "Importing $(basename "$ARCHIVE") into Postgres (this takes ~15–30s) ..."
gunzip -c "$ARCHIVE" | docker compose exec -T postgres psql -U diet_expert -d diet_expert -q

COUNT="$(docker compose exec -T postgres psql -U diet_expert -d diet_expert -t -A \
  -c "SELECT count(*) FROM knowledge_chunks;")"
echo "Done. knowledge_chunks rows: ${COUNT} (expected ~5837)"
if [[ "${COUNT}" -lt 5000 ]]; then
  echo "Warning: row count looks low — check docker compose logs postgres" >&2
  exit 1
fi
