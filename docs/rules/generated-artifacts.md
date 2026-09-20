# Generated Artifact Rules

The controlling rule is: if a tool can regenerate a file, it does not belong in
Git unless a repository rule explicitly defines it as a source artifact.

Never track:

- Vivado project directories and `.xpr` files.
- `.cache`, `.gen`, `.hw`, `.ip_user_files`, `.runs`, `.sim`, or `.Xil` output.
- `vivado_projects/`, `results/`, `src/test/build/`, generated waveforms, logs,
  journals, checkpoints, reports, caches, or temporary IP output.
- `.bit`, `.bin`, `.dcp`, `.ltx`, or `.xsa` build products.
- credentials, private keys, board passwords, licence files, or tokens.

One rule-defined exception exists: a measurement study under
`examples/<example>/docs/<study>/` commits its rendered visualization and its
summary `data/` files even though a script can regenerate them. The study's
conclusions must stay readable without a PyTorch environment, a dataset, or an
hour of compute, and the committed figure is what a reviewer checks the prose
against. The study's input corpus is still never committed.

Commit project-regenerating Tcl under `src/hw/vivado_tcl/<design>/`. Attach a
matching bitstream and hardware handoff to a tagged GitHub Release rather than
Git history. Files staged under `mount/` are deploy payloads and are not
authored source.

After changing `.gitignore`, run:

```bash
git ls-files -i -c --exclude-standard
```

Any output means a tracked file now matches an ignore rule; `.gitignore` does
not untrack it automatically. Review every result without deleting source.
