"""The modules our torchao configs swap in: plain ATen ops, so torch.export
keeps them and the NPU back end recognises the INT8 contraction."""
