"""What every model architecture provides to the three front-end stages.

    load      read the source (default: a Hugging Face safetensors checkpoint)
    quantize  parse the config, resolve the recipe, run the weight plan
    to_mlir   build the ModelGraph from the quantized parameters only

A subclass usually supplies ``parse_config``, ``group_dims``, ``weight_plan``
and ``to_mlir``; the defaults here carry the shared load and quantize steps.
Architectures never import each other, only this module.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from src.export.ir.graph import ModelGraph
from src.export.ir.options import LLMOptions, QuantRecipe, ShapeOptions
from src.export.ir.quantized import QuantizedModel
from src.export.ir.source import SourceModel
from src.export.load import load_weights
from src.export.load.hf import resolve_hf
from src.export.quantize.plan import AwqHook, WeightSpec, run_plan
from src.export.quantize.recipe import choose_group


class Architecture:
    name: str = ''
    title: str = ''                          # used in error messages, e.g. 'SmolLM'
    formats: tuple[str, ...] = ('safetensors',)
    default_source: str = ''

    # Options

    def default_options(self) -> Any:
        return LLMOptions()

    def validate(self, recipe: QuantRecipe, shape: ShapeOptions, awq: AwqHook | None) -> None:
        """Reject options this architecture cannot honour, before anything is read."""

    def describe_options(self, q: QuantizedModel, shape: ShapeOptions) -> dict:
        """What the package records as meta['options']."""
        return LLMOptions.join(q.recipe, shape, q.source.layers).describe()

    # load

    def load(self, source: str, fmt: str, layers: int | None, **kw) -> SourceModel:
        """A local safetensors file or directory, or a Hugging Face cache entry."""
        root = resolve_hf(source)
        tensors = load_weights(root)
        if root.is_file():
            root = root.parent
        config = json.loads((root / 'config.json').read_text())
        return SourceModel(self.name, fmt, source, root, config, tensors, layers)

    # quantize

    def parse_config(self, src: SourceModel) -> Any:
        raise NotImplementedError

    def clip_layers(self, cfg: Any, layers: int) -> None:
        cfg.layers = layers

    def config(self, src: SourceModel) -> Any:
        """The parsed config, with the requested layer limit applied."""
        cfg = self.parse_config(src)
        if src.layers:
            self.clip_layers(cfg, src.layers)
        return cfg

    def group_dims(self, cfg: Any) -> list[int]:
        """Every K dimension the INT4 group size must divide."""
        return []

    def awq_spec(self, cfg: Any):
        """The AWQ FoldSpec, or None when this architecture has no AWQ calibration."""
        return None

    def weight_plan(self, src: SourceModel, cfg: Any) -> list[WeightSpec]:
        raise NotImplementedError

    def quantize(self, src: SourceModel, recipe: QuantRecipe, awq: AwqHook | None = None) -> QuantizedModel:
        cfg = self.config(src)
        recipe = replace(recipe, group=choose_group(recipe.group, self.group_dims(cfg)))
        if recipe.awq and awq is None and (spec := self.awq_spec(cfg)) is not None:
            from src.export.quantize import awq as AWQ
            awq = AWQ.hook_for(src.root, AWQ.calibration_ids(src.root), spec, cfg.layers, recipe.bits, recipe.group)
        params = run_plan(src.tensors, self.weight_plan(src, cfg), recipe, awq)
        return QuantizedModel(src, recipe, cfg, params)

    # to_mlir

    def to_mlir(self, q: QuantizedModel, shape: ShapeOptions) -> ModelGraph:
        raise NotImplementedError


def tokenizer_files(root: Path, *names: str, optional: bool = False) -> dict[str, Path]:
    """Package-relative name -> source path for tokenizer files beside the model."""
    return {name: root / name for name in names if not optional or (root / name).exists()}
