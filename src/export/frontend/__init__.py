"""The export front end: load -> export_graph -> quantize -> to_mlir, all in PyTorch's
own frameworks (transformers, torch.export, torch.fx, torch-mlir).

Its product is ``torch.mlir`` in the torch dialect; the C++ middle end
(src/export/middle-end/) takes it from there.
"""
