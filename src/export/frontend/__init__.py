"""The export front end: load -> quantize -> export_graph -> to_mlir, all in PyTorch's
own frameworks (transformers, torchao, torch.export, torch-mlir).

Its product is ``torch.mlir`` in the torch dialect; the C++ middle end
(src/export/middle-end/) takes it from there.
"""
