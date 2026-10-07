# Environment Rules

Primary language: bash (macOS system bash 3.2 compatible); Dockerfile
Runtime version: macOS 13+ on Apple silicon; Docker Desktop with Rosetta
Package manager: none (image built from references/docker)
Required services: Docker Desktop engine; macOS Screen Sharing (built in)

- Image `npu-vivado:22.04` (linux/amd64); Vivado volume `npu-vivado-tools`
  mounted at `/tools/Xilinx`; container `npu-vivado-gui`.
- VNC listens on `127.0.0.1:5901` only; its password is generated once in
  `~/.config/npu-vivado/vnc-password` and never committed.
- Vivado version: 2026.1, the same as the self-hosted runner
  (`C:\AMDDesignTools\2026.1\Vivado`).
- Deployment uses `deploy-pynq-macos` (board on a direct Ethernet link,
  `xilinx@192.168.2.99`).
