#!/usr/bin/env bash
set -euo pipefail

# Paste your PyPI API token below between the quotes, then run this script.
# Recommended: do NOT paste the token into chat. Alternatively, export PYPI_API_TOKEN in your shell.
PYPI_API_TOKEN="PASTE_YOUR_API_TOKEN_HERE"

if [ -z "${PYPI_API_TOKEN}" ] || [ "${PYPI_API_TOKEN}" = "PASTE_YOUR_API_TOKEN_HERE" ]; then
  echo "Please set PYPI_API_TOKEN inside this script or export it in your environment before running."
  echo "Example: export PYPI_API_TOKEN=\"pypi-...\""
  exit 1
fi

echo "Installing/updating build tools..."
python -m pip install --upgrade build twine

echo "Cleaning previous builds..."
rm -rf dist build
python - <<'PY'
import glob, shutil
for p in glob.glob('*.egg-info'):
    shutil.rmtree(p)
print('cleaned egg-info')
PY

echo "Building sdist and wheel..."
python -m build

echo "Uploading to PyPI..."
python -m twine upload -u __token__ -p "${PYPI_API_TOKEN}" dist/*

echo "Upload finished."
