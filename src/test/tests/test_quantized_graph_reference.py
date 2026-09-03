from __future__ import annotations

import unittest

import numpy as np

from src.test.model.quantized_graph_reference import (
    execute_quantized_graph_reference,
)
from src.test.tests.test_model_runtime import golden, resnet_sequence


class QuantizedGraphReferenceTests(unittest.TestCase):
    def test_vectorized_reference_matches_approved_scalar_operators(self):
        graph = resnet_sequence()
        constants = {
            item.name: np.asarray(item.values, dtype=np.dtype(item.dtype)).reshape(
                item.shape
            )
            for item in graph.constants
        }
        source = np.array([[[[-3], [2]], [[4], [1]]]], dtype=np.int8)
        actual = execute_quantized_graph_reference(
            graph, constants, {"input": source}
        )
        np.testing.assert_array_equal(actual["logits"], golden(source))


if __name__ == "__main__":
    unittest.main()
