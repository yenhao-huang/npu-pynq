# ResNet-18 on the PYNQ-Z1 — Quick Start

This package is self-contained. It carries the notebook, the runtime, the
quantized ResNet-18 model, the ImageNet labels, the demo pictures, and the
`.bit`/`.hwh` overlay the notebook programs. Nothing is downloaded, converted,
or built on the board, and no development host is required.

## Prerequisites

- A PYNQ-Z1 board running the PYNQ image, reachable over the network.
- The board's Jupyter interface, usually `http://192.168.2.99:9090/`.
- About 1.5 hours: one forward pass is 1,814,073,344 MACs on the 8 x 8 systolic
  array and takes roughly an hour.
- Nothing else. No Vivado, no PyTorch, no internet access on the board.

## Three steps

1. **Download and copy.** Take `npu-resnet18-<tag>.zip` from the GitHub Release
   and copy it into the board's notebook directory, for example
   `scp npu-resnet18-<tag>.zip xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/`.
2. **Extract on the board.** `unzip npu-resnet18-<tag>.zip -d npu-resnet18`
   creates one release directory holding `resnet18.ipynb`, `model/`,
   `artifacts/`, `reports/`, and `src/`.
3. **Open `resnet18.ipynb` and run the cells in order.** Pick one of the
   bundled photographs or upload your own, watch the preprocessed tensor, then
   let the inference run. The last cell prints the top-5 ImageNet labels with
   their scores and a CORRECT or INCORRECT verdict.

## Verifying what you received

`package.manifest.json` records the release tag, the source commit, the overlay
digests, the Vivado timing and DRC gates, and the SHA-256 of every file in the
package. `SHA256SUMS` beside the Release asset covers the archive itself. The
board acceptance evidence published with the same Release identifies the same
release tag, source commit, and overlay digests.

## Known limitations

- One forward pass takes roughly an hour; this is a demonstration, not an
  interactive classifier.
- The bundled pictures are five Creative Commons photographs, not ImageNet
  dataset images, so a correct prediction is a demonstration and not an
  accuracy measurement.
- The notebook requires the 8 x 8 overlay with `MAX_K=256` and rejects other
  array dimensions.
- The pinned TorchVision checkpoint is not redistributed in this package; the
  exported INT8 model is.
