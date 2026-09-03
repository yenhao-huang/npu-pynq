import unittest

from src.export.planner import MemoryPlanningError, plan_memory
from src.model.numeric import INT32_MAX
from src.model.resnet import (
    Quantization,
    QuantizedGraph,
    Relu,
    ResidualAdd,
    TensorSpec,
)


Q = Quantization(INT32_MAX, 0, 0)


def tensor(name, shape=(1, 1, 1, 16)):
    return TensorSpec(name, shape, "NHWC", Q)


def residual_graph(tensor_order=None):
    tensors = (
        tensor("input"),
        tensor("skip"),
        tensor("temporary"),
        tensor("output"),
    )
    if tensor_order is not None:
        by_name = {item.name: item for item in tensors}
        tensors = tuple(by_name[name] for name in tensor_order)
    return QuantizedGraph(
        tensors=tensors,
        constants=(),
        commands=(
            Relu("make_skip", "input", "skip"),
            Relu("make_temporary", "skip", "temporary"),
            ResidualAdd("add", "skip", "temporary", "output"),
        ),
        inputs=("input",),
        outputs=("output",),
    )


class MemoryPlannerTests(unittest.TestCase):
    def test_residual_lifetime_and_safe_reuse(self):
        plan = plan_memory(residual_graph())
        input_range = plan.allocation("input")
        skip_range = plan.allocation("skip")
        temporary_range = plan.allocation("temporary")
        output_range = plan.allocation("output")

        self.assertEqual(input_range.offset, temporary_range.offset)
        self.assertNotEqual(skip_range.offset, temporary_range.offset)
        self.assertNotEqual(skip_range.offset, output_range.offset)
        self.assertNotEqual(temporary_range.offset, output_range.offset)
        for allocation in plan.allocations:
            self.assertEqual(allocation.offset % 64, 0)
            self.assertEqual(allocation.allocated_bytes % 64, 0)
        self.assertEqual(skip_range.last_use, 2)

    def test_tensor_declaration_order_does_not_change_plan(self):
        original = plan_memory(residual_graph())
        reordered = plan_memory(
            residual_graph(("output", "temporary", "input", "skip"))
        )
        self.assertEqual(original, reordered)

    def test_capacity_and_size_overflow_fail_explicitly(self):
        plan = plan_memory(residual_graph())
        with self.assertRaisesRegex(MemoryPlanningError, "requires"):
            plan_memory(residual_graph(), arena_limit_bytes=plan.arena_bytes - 1)

        huge = QuantizedGraph(
            tensors=(
                tensor("input", (1, (1 << 31) - 1, (1 << 31) - 1, 3)),
            ),
            constants=(),
            commands=(),
            inputs=("input",),
            outputs=("input",),
        )
        with self.assertRaisesRegex(MemoryPlanningError, "overflow"):
            plan_memory(huge)

    def test_invalid_capacity_type_fails(self):
        with self.assertRaises(TypeError):
            plan_memory(residual_graph(), arena_limit_bytes=True)
        with self.assertRaises(ValueError):
            plan_memory(residual_graph(), arena_limit_bytes=0)


if __name__ == "__main__":
    unittest.main()
