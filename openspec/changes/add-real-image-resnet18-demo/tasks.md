## 1. Preprocessing Contract and Tests

- [x] 1.1 Add tests for Q1.31 scale recovery, ImageNet normalization, symmetric INT8 quantization with round-half-away-from-zero and saturation, the dequantized preview, class-list loading, and top-k decoding.
- [x] 1.2 Implement `src/export/imagenet.py` with the recorded preprocessing contract, lazy Pillow import, and JSON-ready prediction records; verify the focused suite passes.

## 2. Pinned Demo Assets

- [x] 2.1 Add `examples/resnet18/demo-source.json` pinning the class list and one redistributable sample photograph by commit, byte length, SHA-256, and SPDX license.
- [x] 2.2 Add `scripts/download_demo_assets.py` with an approved-host, pinned-revision, fail-closed downloader; verify unapproved hosts, unpinned revisions, non-canonical metadata, missing kinds, and digest mismatches publish nothing.
- [x] 2.3 Verify the downloaded assets land only in the ignored model workspace and leave `git status` clean.

## 3. Demo Input Preparation

- [x] 3.1 Add `scripts/host_reference.py` and move `HostMatrixBackend` and the digest and canonical-write helpers there; verify `verify_model.py` still passes its suite through the shared module.
- [x] 3.2 Add `scripts/prepare_demo_image.py` accepting `--image` and `--expected-class`, running the independent integer reference and `NPUModelRuntime`, and publishing `resnet18.demo.npy` and `resnet18.demo.json` write-once.
- [x] 3.3 Run preparation on the pinned sample; verify the reference and the production host path agree exactly and record the host top-5 and verdict.

## 4. Notebook and Runbook

- [x] 4.1 Add notebook Steps 9 to 12: validate the demo record and its digests, display the original image beside the dequantized NPU input, execute the real image on the board, compare captures, and decode and judge the top-5.
- [x] 4.2 Record the real-image result in the assembled evidence and renumber the review and approval steps; verify the notebook stays output-free and the synthetic path is unchanged.
- [x] 4.3 Document the preparation step, the user-supplied image path, the licensing boundary, and the new notebook steps in the example README and the filetree rules.

## 5. Validation and Handoff

- [x] 5.1 Run the full `src/test/tests` suite and the focused example suites; record exact counts and commands.
- [x] 5.2 Deploy a release to the PYNQ-Z1 and verify every deployed asset digest
      matches this checkout. Release `resnet18-realimage-issue49-20260920-043609`
      carries `deployed_source_commit` 7aaf3bb; the notebook, demo tensor, demo
      record, photograph, class list, model payload/manifest, and
      `src/export/imagenet.py` all match byte for byte. Step 10 was executed on
      the board and rendered `board-step10-preview.png`.
- [x] 5.3 Execute the notebook on the board and obtain human confirmation. The
      agent cannot perform this step: the PL needs root (`/dev/mem`, `/dev/uio*`,
      and the fpga_manager firmware node are root-only) and the board Jupyter
      server is password protected. The human operator ran the notebook on the
      PYNQ-Z1 and confirmed the result; that confirmation, not any agent claim,
      is the evidence for this task.

## 6. Live Demo Notebook

- [x] 6.1 Add the six-step demonstration notebook that programs the overlay,
      takes a picture, preprocesses it on the board, runs it on the 8 x 8 array,
      and plots the top-5, with no digest comparison and no evidence write.
- [x] 6.2 Pin five Creative Commons photographs in `gallery-source.json` and add
      the fail-closed `scripts/download_gallery.py`; verify unapproved hosts,
      digest mismatches, and non-canonical metadata publish nothing. Images were
      selected by running candidates through the exported INT8 model and keeping
      only those it predicts correctly.
- [x] 6.3 Offer the gallery as a Step 3 dropdown beside the upload widget, show
      licence and attribution in the Step 4 caption, and print CORRECT or
      INCORRECT in Step 6 for a picture whose class is declared.
- [x] 6.4 Verify notebook structure mechanically (every code cell parses, no
      markdown in code cells, step headings once each in order) and execute the
      notebook's own cell sources on the board for the four steps that do not
      need root.

## 7. Replace the Notebook With the Demonstration Path

- [x] 7.1 Delete the fourteen-step acceptance notebook and rename the
      demonstration notebook to `examples/resnet18/resnet18.ipynb`, so the
      deployed notebook is the demo. Requested by the repository owner after
      board verification.
- [x] 7.2 Restore the 8 x 8 overlay assertion the acceptance notebook carried,
      and rewrite the notebook intro to point at `run_on_board.py` for
      deterministic numerical acceptance.
- [x] 7.3 Rewrite README section 9 for the demonstration flow, fold the former
      section 10 into it, and keep the `run_on_board.py` acceptance route and
      its `PASS [physical-pynq-z1]` markers documented under it.
- [x] 7.4 Retarget the notebook tests at the demonstration contract: structure,
      ordering, image-before-inference, decoded verdict, physical-overlay-only,
      gallery and upload, and an explicit assertion that the notebook claims no
      acceptance evidence.
