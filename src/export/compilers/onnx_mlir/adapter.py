"""Read imported ONNX IR using the existing MLIR toolchain's parser."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

from src.export.compilers.mlir import toolchain


SOURCE = Path(__file__).with_name('mlir')
BUILD = Path(os.environ.get('NPU_ONNX_ADAPTER_BUILD', 'build/onnx-mlir/adapter-llvm22'))


def executable() -> Path:
    binary = BUILD.resolve() / 'onnx-import-adapter'
    sources = [SOURCE / 'CMakeLists.txt', *SOURCE.rglob('*.cpp')]
    if binary.is_file() and binary.stat().st_mtime >= max(path.stat().st_mtime for path in sources):
        return binary
    compiler_bin = toolchain.llvm_bin()
    toolchain.run(['cmake', '--fresh', '-G', 'Ninja', '-S', str(SOURCE), '-B', str(BUILD.resolve()),
                   f'-DMLIR_DIR={compiler_bin.parent / "lib/cmake/mlir"}',
                   '-DCMAKE_BUILD_TYPE=Release',
                   f'-DCMAKE_CXX_COMPILER={compiler_bin / "clang++"}',
                   f'-DCMAKE_C_COMPILER={compiler_bin / "clang"}'], 'configuring ONNX IR adapter')
    toolchain.run(['cmake', '--build', str(BUILD.resolve()), '-j1'], 'building ONNX IR adapter')
    return binary


def inspect_ir(module: str | Path, output: str | Path) -> dict:
    module = Path(module).resolve(strict=True)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    toolchain.run([str(executable()), str(module), str(output)], 'reading imported ONNX IR')
    graph = json.loads(output.read_text(encoding='utf-8'))
    if graph.get('format') != 'onnx-import-graph/1' or not graph.get('functions'):
        raise ValueError('Imported ONNX IR contains no supported function graph')
    return graph


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_imported(source, directory=None):
    """Import before specialization, or verify a previously prepared import."""
    from .pipeline import prepare_import
    source_hash = file_digest(source)
    directory = Path(directory) if directory else Path('build/onnx-mlir/imports') / source_hash
    if not (directory / 'import.json').is_file():
        prepare_import(source, directory)
    report = json.loads((directory / 'import.json').read_text(encoding='utf-8'))
    if report.get('status') != 'imported' or report.get('source_sha256') != source_hash:
        raise ValueError('ONNX import must succeed and match the source SHA-256')
    artifact = report.get('artifact')
    if artifact != 'model.onnx.mlir':
        raise ValueError('Unexpected imported ONNX artifact name')
    module = directory / artifact
    if not report.get('artifact_sha256') or file_digest(module) != report['artifact_sha256']:
        raise ValueError('Imported ONNX artifact identity mismatch; prepare a fresh import')
    graph = inspect_ir(module, directory / 'graph.json')
    return graph, report, directory
