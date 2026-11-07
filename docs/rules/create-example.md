# File list

- `README.md`: Linux deployment manual. On Windows it runs in WSL, which the
  manual says how to open. There is no separate Windows or macOS manual.
- `<name>.ipynb` provides a runnable demo.

# Notes

1. The key files maintained in Git within the example directory, without affecting the main repository, are listed in the file list above.
2. README instructions use only `src/` and external tools to deploy and run. Do not write custom scripts (e.g., `.py` files).
3. Commit notebooks without outputs. Do not commit generated artifacts.
4. Keep deployment-directory preparation as minimal as possible.
5. If deployment is not possible, explain why. Clearly state when the example has not been run on the board.
6. Keep README instructions as concise as possible.
