#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python3.11 이 필요합니다. 예: brew install python@3.11"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

mkdir -p \
  "${ROOT_DIR}/outputs/logs" \
  "${ROOT_DIR}/outputs/pcd" \
  "${ROOT_DIR}/outputs/debug"

echo "설치 완료"
echo "다음 단계:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python run_demo.py --check-env"
echo "  python run_demo.py"
