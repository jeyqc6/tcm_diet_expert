#!/usr/bin/env bash
# Maintainer: dump local knowledge_chunks and upload to a GitHub Release.
# Requires: psql, gzip, gh auth login
# Usage (from repo root):
#   ./scripts/publish_knowledge_release.sh
#   KB_RELEASE_TAG=v0.1.0-kb PGDATABASE=diet_expert ./scripts/publish_knowledge_release.sh
set -euo pipefail

TAG="${KB_RELEASE_TAG:-v0.1.0-kb}"
PGDATABASE="${PGDATABASE:-diet_expert}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/data/knowledge_chunks.sql.gz"
TITLE="Knowledge base snapshot (${TAG})"

cd "$ROOT"
mkdir -p data

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI: brew install gh && gh auth login" >&2
  exit 1
fi
gh auth status >/dev/null

echo "Dumping knowledge_chunks from database ${PGDATABASE} ..."
pg_dump -d "${PGDATABASE}" -t knowledge_chunks --data-only --no-owner | gzip -c > "${OUT}"
ls -lh "${OUT}"

ROWS="$(psql -d "${PGDATABASE}" -t -A -c "SELECT count(*) FROM knowledge_chunks;")"
NOTES="$(cat <<EOF
Pre-built BGE-M3 embeddings for \`knowledge_chunks\` (${ROWS} rows, ~93 MB in Postgres, ~34 MB download).

**Import (after \`docker compose up -d postgres\`):**

\`\`\`bash
./scripts/import_knowledge_chunks.sh
\`\`\`

Or manually:

\`\`\`bash
curl -fL -o knowledge_chunks.sql.gz \\
  https://github.com/jeyqc6/tcm_diet_expert/releases/download/${TAG}/knowledge_chunks.sql.gz
gunzip -c knowledge_chunks.sql.gz | docker compose exec -T postgres psql -U diet_expert -d diet_expert
\`\`\`

Not included in git — regenerate with \`scripts/publish_knowledge_release.sh\` when the KB changes.
EOF
)"

if gh release view "${TAG}" >/dev/null 2>&1; then
  echo "Release ${TAG} exists — uploading asset ..."
  gh release upload "${TAG}" "${OUT}" --clobber
else
  echo "Creating release ${TAG} ..."
  gh release create "${TAG}" "${OUT}" --title "${TITLE}" --notes "${NOTES}"
fi

echo "Published: https://github.com/jeyqc6/tcm_diet_expert/releases/tag/${TAG}"
