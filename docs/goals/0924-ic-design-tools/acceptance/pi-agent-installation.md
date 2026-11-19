# Install pi-agent

pi-agent is installed with **npm**. Python venvs manage the IC Python tools,
not pi itself. Run these commands in Linux/WSL Bash.

## 1. Reuse the existing installation

The experiment container `b730c8d699af` already has pi **0.74.2** installed
under howard's NVM Node **20.20.2** environment:

```text
~/.nvm/versions/node/v20.20.2/bin/pi
~/.nvm/versions/node/v20.20.2/lib/node_modules/@earendil-works/pi-coding-agent
```

Use it without reinstalling or activating a Python venv:

```bash
export PATH="$HOME/.nvm/versions/node/v20.20.2/bin:$PATH"
unset npm_config_prefix
node --version
pi --version
npm root -g
cd /workspace/npu/worktrees/npu-issue74-a
pi
```

Expected versions: Node `v20.20.2`, pi `0.74.2`. The npm root should be the
NVM directory above. These paths and versions were verified in the container;
they are not paths to assume on another machine.

## 2. Install on a new environment

Use Node.js 20.11+ and npm; Node 20.20.2 is the tested baseline. If NVM is
already installed, select that version first:

```bash
nvm install 20.20.2
nvm use 20.20.2
```

Install the tested pi version with npm:

```bash
npm install --global @earendil-works/pi-coding-agent@0.74.2
command -v pi
pi --version
```

With NVM, this installs into the selected Node version's prefix. Switching
Node versions can change which global packages are available.

If using a system Node installation without a writable npm prefix, use a
user-owned prefix instead:

```bash
export npm_config_prefix="$HOME/.local"
export PATH="$npm_config_prefix/bin:$PATH"
npm install --global @earendil-works/pi-coding-agent@0.74.2
pi --version
```

Keep the chosen PATH and prefix settings in later terminals. Start `pi` from
your project root and configure a model provider through pi's login/settings
before requesting model responses. Installing pi does not configure credentials.

## 3. Connect pi to IC tools

From your project root, install the published
[npm package](https://www.npmjs.com/package/@jony2156/ai-eda-tools):

```bash
pi install npm:@jony2156/ai-eda-tools@0.1.0 --local
pi
```

`--local` saves the npm package declaration in this project's settings.
Pi loads the package's declared extension; no `ic-tools init pi` is needed.
The managed runtime is prepared automatically. When migrating from the old
wrapper, follow [pi-agent acceptance](pi-agent.md#1-install-with-pi) to avoid
registering the tools twice. Run `/ic` inside pi to check the backends.
