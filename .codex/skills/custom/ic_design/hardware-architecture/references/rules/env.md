# Environment Rules

This skill edits an SVG source file and uses the repository's existing RTL,
Vivado Tcl, and Markdown files. It requires no board connection, synthesis,
package installation, or network service.

Use an available XML parser to verify the SVG. Use an available SVG renderer
to create a temporary preview at the original `viewBox` dimensions, then
inspect the rendered result. Prefer tools already present in the workspace;
do not install a new renderer solely for a label-only change.

Do not assume a particular operating system path for Python, Node.js, or the
renderer. Keep generated preview files under an ignored build directory.
