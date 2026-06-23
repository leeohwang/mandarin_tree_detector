#!/usr/bin/env bash
# =============================================================================
# Grove — one-command LOCAL setup (Mac, no GPU).
# Creates a virtualenv and installs the GPU-free [review] extra so you can run
# the review UI. This NEVER installs torch/autodistill (those live on Kaggle).
# Usage:  ./setup.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Grove local setup"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON' not found. Install Python 3.10+ and retry (or set PYTHON=...)." >&2
  exit 1
fi

# Require Python >= 3.10
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"ERROR: Python 3.10+ required, found {sys.version.split()[0]}")
PY

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtualenv in $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# NOTE: we deliberately do NOT `pip install --upgrade pip` here. On a flaky
# network (e.g. behind a firewall/proxy) a half-finished pip self-upgrade can
# leave the venv's pip corrupted ("No module named 'pip._vendor.urllib3'"),
# which then blocks every install. The pip that ships with `python -m venv` is
# perfectly capable of installing the [review] extra, so we just use it.
echo "==> Installing grove with the [review] extra (GPU-free)"
if ! python -m pip install -e ".[review,dev]"; then
  # Self-heal for a firewalled / TLS-intercepting network (e.g. a corporate or
  # VPN proxy that re-signs HTTPS): pip's bundled cert store won't trust the
  # proxy's CA, but the macOS keychain does. Export the keychain trust store to
  # a PEM and point pip at it, and use a domestic mirror with a longer timeout
  # so large wheels don't time out across the link. Override the mirror with
  # `PIP_INDEX_URL=... ./setup.sh` if you prefer a different one.
  echo "==> Direct install failed; retrying via the macOS keychain trust store + mirror..."
  CA_BUNDLE="$VENV_DIR/macos-ca.pem"
  : > "$CA_BUNDLE"
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >> "$CA_BUNDLE" 2>/dev/null || true
  security find-certificate -a -p /Library/Keychains/System.keychain                         >> "$CA_BUNDLE" 2>/dev/null || true
  security find-certificate -a -p "$HOME/Library/Keychains/login.keychain-db"                >> "$CA_BUNDLE" 2>/dev/null || true
  INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  SSL_CERT_FILE="$CA_BUNDLE" REQUESTS_CA_BUNDLE="$CA_BUNDLE" PIP_CERT="$CA_BUNDLE" \
    python -m pip install -e ".[review,dev]" -i "$INDEX_URL" --timeout 120 --retries 5
fi

# Robustness: some Python builds do not reliably process the .pth import hook that
# setuptools' editable install relies on, leaving `import grove` broken (symptom:
# "ModuleNotFoundError: No module named 'grove'" when running `grove`/`make review`).
# Detect that from a neutral CWD and, if needed, symlink the package straight into
# site-packages. Python's normal package finder always scans site-packages, so this
# does not depend on .pth processing at all and is stable across runs.
if ! ( cd / && python -c "import grove" >/dev/null 2>&1 ); then
  echo "==> Editable import hook inactive on this Python; linking grove into site-packages"
  SITEDIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  ln -snf "$PWD/grove" "$SITEDIR/grove"
  ( cd / && python -c "import grove; print('   grove now importable from', grove.__file__)" )
fi

# Seed a config.yaml from the template if the user has none yet.
if [ ! -f config.yaml ]; then
  echo "==> Creating config.yaml from config.example.yaml (edit the paths inside)"
  cp config.example.yaml config.yaml
fi

cat <<'DONE'

==> Done.
   Next:
     1. Put a labeled dataset where config.yaml points (paths.export_dir).
     2. Run the review UI:   make review     (or: .venv/bin/grove review --config config.yaml)

   See OPERATOR_GUIDE.md for the full, minimal run checklist.
DONE
