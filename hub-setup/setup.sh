#!/usr/bin/env bash
# One-time setup on the tutorial server (The Littlest JupyterHub): run with  sudo bash setup.sh
#   master copies  <root>/master/mosaic, <root>/master/shasta   read-only, only an admin changes them
#   scores         <root>/scores/mosaic                          the one shared folder users can write to
#   each user gets their own ~/mosaic and ~/shasta when their Jupyter server starts (smc-sync)
set -euo pipefail
ROOT=${SMC_ROOT:-/home/smc-tutorial}
GROUP=${SMC_GROUP:-jupyterhub-users}
USERENV=${SMC_USERENV:-/opt/tljh/user}
HERE=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date +%Y%m%d-%H%M%S)
[ "$(id -u)" = 0 ] || { echo "Run this with sudo."; exit 1; }

# 1. The master copies: what is in the shared folders now, without run leftovers. The old folders are backed up first.
mkdir -p "$ROOT/master" "$ROOT/_backup"
for t in mosaic shasta; do
  if [ -d "$ROOT/$t" ] && [ ! -d "$ROOT/master/$t" ]; then
    cp -a "$ROOT/$t" "$ROOT/_backup/$t-$STAMP"
    rsync -a --exclude='.ipynb_checkpoints' --exclude='__pycache__' --exclude='scores' --exclude='assets' \
          --exclude='*.xdf' --exclude='*.osm' "$ROOT/$t/" "$ROOT/master/$t/"
    rm -rf "$ROOT/$t"
  fi
done
for nb in "$ROOT"/master/*/*.ipynb; do                      # the masters start clean, without saved output
  "$USERENV/bin/jupyter" nbconvert --clear-output --inplace "$nb" >/dev/null 2>&1 || true
done
chown -R root:root "$ROOT/master" "$ROOT/_backup"
chmod -R u=rwX,go=rX "$ROOT/master"
chmod 700 "$ROOT/_backup"

# 2. Scores: the only place users write. Setgid keeps the group, and the sticky bit stops one user deleting another's file.
mkdir -p "$ROOT/scores/mosaic"
chown -R root:"$GROUP" "$ROOT/scores"
chmod 3775 "$ROOT/scores" "$ROOT/scores/mosaic"
chown root:root "$ROOT"; chmod 755 "$ROOT"                  # users can no longer rename or delete things in the top folder

# 3. The command that gives a user their copy, and the hook that runs it when a user's server starts.
install -m 755 "$HERE/smc-sync" /usr/local/bin/smc-sync
ln -sf /usr/local/bin/smc-sync "$USERENV/bin/smc-sync"      # /usr/local/bin is not on the PATH of a user's terminal; this folder is
CONFIG="$USERENV/etc/jupyter/jupyter_server_config.py"
mkdir -p "$(dirname "$CONFIG")"
grep -q "smc-sync" "$CONFIG" 2>/dev/null || cat >> "$CONFIG" <<'PY'

# SMC tutorial: give each user their own copy of the tutorials when their server starts
import subprocess
try:
    subprocess.run(["/usr/local/bin/smc-sync"], timeout=60)
except Exception:
    pass
PY

echo; echo "Done. Masters:"; ls -l "$ROOT/master"; echo "Scores:"; ls -ld "$ROOT/scores" "$ROOT/scores/mosaic"
echo "Test it as a user:  sudo -u jupyter-bennett /usr/local/bin/smc-sync && ls /home/jupyter-bennett"
