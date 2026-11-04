"""The export front end: load -> quantize -> export_graph -> to_mlir, all in PyTorch's
own frameworks (transformers, torchao, torch.export, torch-mlir).

Its product is ``model.mlir`` in upstream dialects (linalg on tensors); the
C++ back end (src/export/backend/) takes it from there.
"""
