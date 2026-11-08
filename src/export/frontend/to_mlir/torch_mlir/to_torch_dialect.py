"""ExportedPrograms -> one MLIR module in the torch dialect.

Every program becomes a function of the same module; parameters and buffers
are frozen into ``torch.vtensor.literal`` constants (dense_resource blobs),
shared between functions when they are the same tensor.
"""

from __future__ import annotations

import torch


def to_torch_dialect(programs: dict[str, torch.export.ExportedProgram]):
    """Return the torch-dialect ``torch_mlir.ir.Module``."""
    from torch_mlir import ir
    from torch_mlir.dialects import torch as torch_d
    from torch_mlir.extras.fx_importer import FxImporter

    context = ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    for name, ep in programs.items():
        # Non-persistent buffers (RoPE's inv_freq) live in ep.constants, where
        # the frozen import does not look for buffers.
        # _state_dict is a private torch attribute, so recheck this on a torch upgrade.
        ep._state_dict.update({n: ep.constants[n] for n in ep.graph_signature.inputs_to_buffers.values()
                               if n in ep.constants and n not in ep.state_dict})
        # This name is part of the ABI: the runtime calls _mlir_ciface_<name>.
        importer.import_frozen_program(ep, func_name=name)
    return importer.module
