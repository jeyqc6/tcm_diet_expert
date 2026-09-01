#!/usr/bin/env bash
# One-shot deploy on a fresh Ubuntu VPS (Oracle Cloud Always Free ARM, etc.).
# Run on the server as a normal user with sudo:
#   curl -fsSL https://raw.githubusercontent.com/jeyqc6/tcm_diet_expert/main/scripts/vps_deploy.sh | bash
# Or from a cloned repo:
#   ./scripts/vps_deploy.sh
#
# Prerequisites (manual, in Oracle Cloud console):
#   - VM: Ubuntu 22.04/24.04 ARM (Ampere), >= 4 GB RAM recommended (torch+BGE-M3)
#   - Ingress / security list: TCP 22, 3000, 8123 from 0.0.0.0/0 (tighten later)
#   - Edit .env with at least one LLM provider key before chatting
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jeyqc6/tcm_diet_expert.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/tcm_diet_expert}"
KB_TAG="${KB_RELEASE_TAG:-v0.1.0-kb}"

detect_public_ip() {
  curl -fsS --max-time 5 https://ifconfig.me 2>/dev/null \
    || curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null \
    || true
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing Docker (needs sudo) ..."
  sudo apt-get update -qq
  sudo apt-get install -y docker.io docker-compose-v2 git curl
  sudo systemctl enable --now docker
  sudo usermod -aG docker "$USER" || true
  if ! docker ps >/dev/null 2>&1; then
    echo "Docker installed. Log out and SSH back in, then re-run this script." >&2
    exit 1
  fi
}

upsert_env_var() {
  local file="$1" key="$2" value="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    sed -i.bak "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '\n%s=%s\n' "$key" "$value" >>"$file"
  fi
}

echo "==> Checking Docker ..."
ensure_docker

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  echo "==> Cloning into ${INSTALL_DIR} ..."
  git clone "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"
git pull --ff-only || true

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit LLM keys before using chat:"
  echo "  nano ${INSTALL_DIR}/.env"
fi

PUBLIC_IP="$(detect_public_ip)"
if [[ -z "${PUBLIC_API_BASE_URL:-}" && -n "$PUBLIC_IP" ]]; then
  export PUBLIC_API_BASE_URL="http://${PUBLIC_IP}:8123"
  upsert_env_var .env PUBLIC_API_BASE_URL "$PUBLIC_API_BASE_URL"
  echo "Set PUBLIC_API_BASE_URL=${PUBLIC_API_BASE_URL}"
elif [[ -n "${PUBLIC_API_BASE_URL:-}" ]]; then
  upsert_env_var .env PUBLIC_API_BASE_URL "$PUBLIC_API_BASE_URL"
else
  echo "Could not detect public IP; set PUBLIC_API_BASE_URL in .env manually." >&2
  exit 1
fi

echo "==> Building and starting stack (first run may take 15–30 min: torch image + BGE download) ..."
docker compose up --build -d

echo "==> Importing knowledge snapshot (${KB_TAG}) ..."
chmod +x scripts/import_knowledge_chunks.sh
KB_RELEASE_TAG="$KB_TAG" ./scripts/import_knowledge_chunks.sh

echo ""
echo "Deploy finished."
echo "  Frontend:  http://${PUBLIC_IP}:3000"
echo "  API:       http://${PUBLIC_IP}:8123/healthz"
echo ""
echo "If pages load but chat fails, check LLM keys in ${INSTALL_DIR}/.env"
echo "Logs: docker compose -f ${INSTALL_DIR}/docker-compose.yml logs -f api"
