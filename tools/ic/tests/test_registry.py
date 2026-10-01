"""The registry is the single source of truth, so these tests guard the
properties every generated surface depends on."""

from __future__ import annotations

import pytest

from ic_core.registry import CATEGORIES, all_op_names, find_op, iter_ops, load_all


def test_every_category_registers_backends():
    load_all()
    assert set(CATEGORIES) == {"lint", "sim", "debug", "view", "synth"}
    for category in CATEGORIES.values():
        assert category.backends, f"{category.name} has no backend"
        assert category.default_backend in category.backends


def test_op_names_are_globally_unique():
    """Op names double as MCP tool names and CLI subcommands, so a collision
    would silently shadow a tool rather than fail loudly."""
    names = all_op_names()
    assert len(names) == len(set(names)), names


def test_schemas_are_generatable():
    """Every surface renders these; an unserialisable model breaks all of them."""
    for _category, op in iter_ops():
        assert op.In.model_json_schema()["type"] == "object"
        assert op.Out.model_json_schema()["type"] == "object"


def test_every_op_has_a_described_input():
    """Field descriptions are the only documentation an agent sees."""
    for _category, op in iter_ops():
        assert op.summary.strip()
        for name, field in op.In.model_fields.items():
            assert field.description, f"{op.name}.{name} has no description"


def test_backends_implement_their_ops():
    load_all()
    for category in CATEGORIES.values():
        for name, entry in category.backends.items():
            impl = entry.impl()
            for op in category.ops:
                assert hasattr(impl, op.name), f"{category.name}/{name} lacks {op.name}"


def test_unknown_op_is_rejected():
    from ic_core.errors import UnknownOp

    with pytest.raises(UnknownOp):
        find_op("no_such_tool")
