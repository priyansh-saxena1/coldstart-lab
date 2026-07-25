#!/usr/bin/env bash
# colab_setup.sh — unzip + install coldstart-lab in a Colab session, robustly.
#
# Why this exists: the naive `!unzip x.zip && cd x && pip install -e .` breaks in
# several ways that all look like "it just doesn't work":
#
#   * The archive may unzip to ./coldstart-lab/, or to ./ directly, or to a
#     nested ./coldstart-lab/coldstart-lab/ if it was re-zipped. We locate the
#     directory that actually contains pyproject.toml instead of assuming.
#   * macOS zips carry a __MACOSX/ sidecar full of ._ files that confuse both
#     the search above and setuptools. We delete it.
#   * Re-running after a re-upload leaves a stale tree that shadows the new one.
#     We remove the old extraction first.
#   * `cd` inside a notebook `!` cell does not persist; the install must happen
#     in the same shell invocation as the cd. This script does both.
#
# Usage in Colab:
#     !bash colab_setup.sh            # if the script is already extracted
#     # or, from a bare upload:
#     !unzip -o -q coldstart-lab.zip && bash coldstart-lab/scripts/colab_setup.sh
set -euo pipefail

ZIP="${1:-/content/coldstart-lab.zip}"
DEST="${2:-/content}"

echo "==> looking for archive: $ZIP"
if [[ -f "$ZIP" ]]; then
  # Remove any previous extraction so a re-upload actually takes effect.
  rm -rf "$DEST/coldstart-lab" "$DEST/__MACOSX"
  echo "==> extracting"
  unzip -o -q "$ZIP" -d "$DEST"
  rm -rf "$DEST/__MACOSX"
else
  echo "    (no zip at that path; assuming the repo is already extracted)"
fi

# Find the real project root: the shallowest directory holding pyproject.toml.
ROOT="$(find "$DEST" -maxdepth 3 -name pyproject.toml -not -path '*/__MACOSX/*' \
        -printf '%d %h\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)"

if [[ -z "${ROOT:-}" ]]; then
  echo "ERROR: could not find pyproject.toml under $DEST" >&2
  echo "       Contents of $DEST:" >&2
  ls -la "$DEST" >&2
  exit 1
fi

echo "==> project root: $ROOT"
cd "$ROOT"

echo "==> installing (core + distributed + plots)"
# Some managed Python environments (PEP 668) refuse a plain pip install into the
# system interpreter. Colab does not, but retrying with the override keeps this
# script working on Kaggle/Codespaces/local boxes too.
pip install -q -e ".[distributed,plots]" \
  || pip install -q -e ".[distributed,plots]" --break-system-packages

echo "==> verifying"
python - <<'PY'
import coldstart_lab
from coldstart_lab.models import MODEL_REGISTRY
from coldstart_lab.distributed import Coordinator  # noqa: F401
print(f"coldstart_lab {coldstart_lab.__version__} OK "
      f"({len(MODEL_REGISTRY)} models registered, distributed available)")
PY

echo
echo "==> ready. Project root: $ROOT"
echo "    Run a single-machine benchmark:"
echo "      cd $ROOT && coldstart-lab --model smollm2-135m --device cpu --out-dir ./out"
echo "    Or join the fleet (needs COLDSTART_DB_URL):"
echo "      cd $ROOT && coldstart-fleet work --device cuda --device-class t4"
