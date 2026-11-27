#!/bin/sh
# Makes the sign recogniser start by itself whenever the Pi gets power (no screen, no login).
# Run once from the project folder:   sh deploy/install_autostart.sh
set -e
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "No .venv here ($(pwd)). Do the setup in requirements-pi.txt first." >&2
  exit 1
fi
if [ "$(pwd)" != "$HOME/signlng_cam" ]; then
  echo "The project must be at $HOME/signlng_cam (it is at $(pwd))." >&2
  exit 1
fi
mkdir -p "$HOME/.config/systemd/user"
cp deploy/signlang.service "$HOME/.config/systemd/user/signlang.service"
# start the user's services at boot even when nobody logs in
# (allowed for your own user without a password on Raspberry Pi OS; sudo only as a fallback)
loginctl enable-linger "$USER" 2>/dev/null || sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable signlang.service
systemctl --user restart signlang.service
echo
echo "Installed. It starts automatically at every boot and says \"ready\" when it is listening."
echo "  status / live log : journalctl --user -u signlang -f"
echo "  stop (e.g. to run run.py by hand, or record signs): systemctl --user stop signlang"
echo "  start again       : systemctl --user start signlang"
echo "  turn autostart off: systemctl --user disable --now signlang"
