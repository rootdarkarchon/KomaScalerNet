# Debian 13: Install Codex and Start the Build

These instructions use the official standalone Codex installer, so Node.js is
not required merely to run Codex. Run commands as your normal user except where
`sudo` is shown.

Official references:

- [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)
- [Codex authentication](https://learn.chatgpt.com/docs/auth)
- [Codex sandboxing](https://learn.chatgpt.com/docs/sandboxing)
- [.NET on Debian](https://learn.microsoft.com/en-us/dotnet/core/install/linux-debian)

## 1. Inspect the machine

```bash
cat /etc/os-release
uname -m
nvidia-smi
git --version
```

This handoff expects Debian 13 x86-64, the already-tested NVIDIA driver/CUDA
environment, and the RTX 3060. Stop and report the output if those assumptions
are wrong.

## 2. Install basic tools and Codex sandbox support

```bash
sudo apt update
sudo apt install -y ca-certificates curl git unzip bubblewrap wget
```

The official Codex documentation recommends the distribution `bubblewrap`
package on Linux. It keeps workspace sandboxing reliable.

## 3. Install Codex CLI

Use the interactive official installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

The default command location is `$HOME/.local/bin`. If `codex` is not found in
a new shell, add it to the shell you actually use. For zsh:

```bash
printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.zshrc"
exec zsh
```

For bash, use `$HOME/.bashrc` and `exec bash` instead.

Verify the installation:

```bash
codex --version
codex doctor
```

Do not use `sudo codex`; authentication and configuration belong to your user.

## 4. Sign in

For normal interactive work with a ChatGPT account:

```bash
codex login
codex login status
```

If the machine is headless or the browser callback cannot reach it:

```bash
codex login --device-auth
```

An API key is optional and uses API billing rather than ChatGPT subscription
access. If intentionally using one, never paste it into a prompt or shell
history; place it in an environment variable and pipe it:

```bash
printenv OPENAI_API_KEY | codex login --with-api-key
```

## 5. Install .NET 10 SDK

Debian 13 is supported by .NET 10. Add Microsoft's Debian 13 repository and
install the SDK:

```bash
wget https://packages.microsoft.com/config/debian/13/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb
sudo apt update
sudo apt install -y dotnet-sdk-10.0
dotnet --info
```

Also install libvips development/runtime packages. Package names can vary by
Debian point release, so inspect candidates first:

```bash
apt-cache search '^libvips'
sudo apt install -y libvips-dev libvips-tools
vips --version
```

## 6. Extract the handoff into a new repository

Choose a location on a filesystem with room for builds, a cache, and the seven
local ONNX files:

```bash
mkdir -p "$HOME/src/KomaScaler.Net"
cd "$HOME/src/KomaScaler.Net"
unzip /path/to/KomaScaler.Net-codex-handoff.zip
git init
git status
```

Review `README.md`, `CODEX-PROMPT.md`, and `.codex/config.toml` before trusting
the repository. Do not copy ONNX models into Git history.

## 7. Launch Codex

The build needs NuGet network access but should remain write-limited to the
repository. Start the interactive CLI from the repository root:

```bash
codex \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  -c sandbox_workspace_write.network_access=true \
  "Read CODEX-PROMPT.md and implement the complete application. Follow AGENTS.md, keep the solution green after each milestone, and continue until the acceptance report is produced."
```

Codex may ask permission for system package operations, GPU inspection, or
access outside the repository. Read each request; do not grant broad filesystem
access merely for convenience.

If the session is interrupted, return to the repository and use:

```bash
codex resume
```

Choose the KomaScaler.Net session, then ask it to continue from the current
working tree and acceptance checklist.

## 8. Operator inputs Codex will eventually need

- The seven local FP32 ONNX files listed in
  `models/models.production.json`.
- Read access to the model directory and write access to a cache directory.
- Permission to run opt-in GPU tests and `nvidia-smi`.
- A real manga page you own for final visual/end-to-end validation.
- The Suwayomi configuration location and, if used, a shared token.

Codex can build and run all fake-backend tests before the model files are
installed. When it reaches GPU acceptance, provide their absolute directory
path rather than moving licensed models into the repository.

## Troubleshooting

- `codex: command not found`: confirm `$HOME/.local/bin` is on `PATH`.
- Sandbox warning on Linux: confirm `command -v bwrap` and rerun
  `codex doctor`.
- Browser login cannot complete: use `codex login --device-auth`.
- NuGet restore cannot reach the network: restart with the exact network-enabled
  workspace-write command above.
- CUDA provider unavailable: do not accept CPU fallback. Capture
  `nvidia-smi`, `ldconfig -p | grep -E 'libcuda|libcudnn|libcudart'`, and the
  service logs for diagnosis.
- Resume uncertainty: run `git status`, `dotnet build`, and `dotnet test`, then
  tell Codex to reconcile the working tree with `docs/history/ACCEPTANCE-CRITERIA.md`.
