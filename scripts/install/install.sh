#!/bin/sh
set -eu

publish_directory=${1:-./artifacts/publish}
auth_mode=${KOMASCALER_AUTH_MODE:-authenticated}
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository=$(CDPATH= cd -- "$script_directory/../.." && pwd)
test -x "$publish_directory/komascaler"
case "$auth_mode" in
  authenticated) command -v openssl >/dev/null 2>&1 || { echo "openssl is required for authenticated installation." >&2; exit 1; } ;;
  disabled) ;;
  *) echo "KOMASCALER_AUTH_MODE must be authenticated or disabled." >&2; exit 2 ;;
esac
command -v curl >/dev/null 2>&1 || {
  echo "curl is required by the systemd readiness probe." >&2
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is required by the production TensorRT deployment." >&2
  exit 1
}
command -v ldconfig >/dev/null 2>&1 || {
  echo "ldconfig is required to validate production TensorRT libraries." >&2
  exit 1
}
for library in libnvinfer.so.10 libnvonnxparser.so.10 libcudnn.so.9 libcudart.so.12; do
  ldconfig -p | grep -q "$library" || {
    echo "Required TensorRT runtime library is unavailable: $library" >&2
    echo "Run scripts/engines/tensorrt-preflight.sh before installing." >&2
    exit 1
  }
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null || {
  echo "The NVIDIA driver is not operational; refusing TensorRT-default installation." >&2
  exit 1
}

if ! getent passwd komascaler >/dev/null; then
  useradd --system --home-dir /var/lib/komascaler --shell /usr/sbin/nologin komascaler
fi
install -d -o root -g root -m 0755 /usr/lib/komascaler /etc/komascaler
install -d -o komascaler -g komascaler -m 0750 /var/lib/komascaler /var/lib/komascaler/models /var/cache/komascaler /var/cache/komascaler/tensorrt
cp -a "$publish_directory/." /usr/lib/komascaler/
test -f /usr/lib/komascaler/runtimes/linux-x64/native/libonnxruntime_providers_tensorrt.so || {
  echo "Published ONNX Runtime TensorRT provider library is missing." >&2
  exit 1
}
if ! test -e /etc/komascaler/appsettings.json; then
  install -o root -g root -m 0644 "$repository/config/appsettings.example.json" /etc/komascaler/appsettings.json
else
  echo "Preserving existing /etc/komascaler/appsettings.json."
fi
sh "$script_directory/configure-install-auth.sh" "$auth_mode" \
  "$repository/deploy/systemd/komascaler.service" /etc/systemd/system/komascaler.service \
  /etc/komascaler/token root root
if ! test -e /var/lib/komascaler/models/models.production.json; then
  install -o root -g root -m 0644 "$repository/models/models.production.json" /var/lib/komascaler/models/models.production.json
fi
systemctl daemon-reload
systemctl enable komascaler.service

echo "Copy the seven verified .onnx files to /var/lib/komascaler/models."
if test "$auth_mode" = authenticated; then
  echo "Authentication is enabled with the root-only /etc/komascaler/token systemd credential."
else
  echo "Authentication is disabled; the installed unit has no LoadCredential directive. Bind only to a trusted local interface."
fi
echo "Production uses TensorRT 8/960/960 with PreloadOnStartup=true. An empty engine cache takes about ten minutes to build before readiness."
echo "Generate and validate the production engine cache before starting; see docs/operations/DEPLOYMENT.md and scripts/engines/tensorrt-production-cache.sh."
echo "Otherwise start komascaler and wait for the systemd readiness gate to complete."
echo "Verify readiness with: curl --fail http://127.0.0.1:9999/health/ready"
