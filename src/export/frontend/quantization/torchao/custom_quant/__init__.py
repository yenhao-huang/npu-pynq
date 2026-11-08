"""The modules our torchao configs swap in: plain ATen ops, so torch.export
keeps the INT8 contraction visible to whatever compiles the graph."""
