#!/bin/sh
set -eu

mode=${1:?Usage: configure-install-auth.sh authenticated|disabled unit-source unit-destination token-path [owner] [group]}
unit_source=${2:?Missing unit source}
unit_destination=${3:?Missing unit destination}
token_path=${4:?Missing token path}
owner=${5:-root}
group=${6:-root}

case "$mode" in
  authenticated)
    command -v openssl >/dev/null 2>&1 || { echo "openssl is required to generate the authentication token." >&2; exit 1; }
    install -o "$owner" -g "$group" -m 0644 "$unit_source" "$unit_destination"
    if test ! -f "$token_path"; then
      token_temporary="$token_path.new.$$"
      trap 'unlink "$token_temporary" 2>/dev/null || true' EXIT HUP INT TERM
      umask 077
      openssl rand -hex 32 > "$token_temporary"
      install -o "$owner" -g "$group" -m 0600 "$token_temporary" "$token_path"
      unlink "$token_temporary"
      trap - EXIT HUP INT TERM
      echo "Generated authentication credential: $token_path"
    else
      chown "$owner:$group" "$token_path"
      chmod 0600 "$token_path"
      echo "Preserved existing authentication credential: $token_path"
    fi
    ;;
  disabled)
    unit_temporary="$unit_destination.new.$$"
    trap 'unlink "$unit_temporary" 2>/dev/null || true' EXIT HUP INT TERM
    awk '
      /^LoadCredential=Upscaling__Security__Token:/ { next }
      /^Environment=Upscaling__Security__Token=/ { next }
      /^\[Install\]$/ { print "Environment=Upscaling__Security__Token="; print; next }
      { print }
    ' "$unit_source" > "$unit_temporary"
    install -o "$owner" -g "$group" -m 0644 "$unit_temporary" "$unit_destination"
    unlink "$unit_temporary"
    trap - EXIT HUP INT TERM
    if test -e "$token_path"; then
      unlink "$token_path"
      echo "Removed disabled authentication credential: $token_path"
    fi
    ;;
  *)
    echo "KOMASCALER_AUTH_MODE must be authenticated or disabled." >&2
    exit 2
    ;;
esac
