"""Operator library shared by every model.

Each operator emits MLIR through ``FuncBuilder`` and has a NumPy reference
with identical numerics; models compose operators and never emit MLIR
directly, so a new model only adds a frontend.
"""
