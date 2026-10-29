"""ExportedPrograms -> one MLIR module in the torch dialect.

Every program becomes a function of the same module; parameters and buffers
are frozen into ``torch.vtensor.literal`` constants (dense_resource blobs),
shared between functions when they are the same tensor.
"""

from __future__ import annotations

import torch


def to_torch_dialect(programs: dict[str, torch.export.ExportedProgram]) -> str:
    from iree.compiler.extras.fx_importer import FxImporter
    from iree.compiler.ir import Context

    importer = FxImporter(context=Context())
    for name, ep in programs.items():
        # Non-persistent buffers (RoPE's inv_freq) live in ep.constants, where
        # the frozen import does not look for buffers.
        ep._state_dict.update({n: ep.constants[n] for n in ep.graph_signature.inputs_to_buffers.values()
                               if n in ep.constants and n not in ep.state_dict})
        importer.import_frozen_program(ep, func_name=name)
    return str(importer.module)
