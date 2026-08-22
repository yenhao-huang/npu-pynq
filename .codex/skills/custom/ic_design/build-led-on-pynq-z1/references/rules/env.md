# Environment Rule

Primary languages: Tcl/XDC and Python
Runtime versions: AMD Vivado compatible with PYNQ-Z1; PYNQ Linux image
Package manager: none
Frameworks: PYNQ Python API
Service manager: none
Required services: Jupyter or SSH on a reachable PYNQ-Z1 board

- Use the repository's existing Vivado project and generated artifacts.
- Do not install Vivado, drivers, packages, or licenses unless explicitly asked.
- Keep `.bit` and `.hwh` files paired with the same basename for deployment.
- Never store board passwords or SSH private keys in the skill.
