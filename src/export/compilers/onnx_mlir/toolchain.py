"""Invoke ONNX-MLIR at the import boundary with retained diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import uuid
import hashlib


PIN = json.loads(Path(__file__).with_name('toolchain.json').read_text(encoding='utf-8'))


class OnnxMlirError(RuntimeError):
    """The compiler could not produce a fresh ONNX dialect module."""


@dataclass(frozen=True)
class Compiler:
    binary: str | None = None
    timeout: int = 300

    def command(self, source: Path, output: Path) -> list[str]:
        source = source.resolve()
        output = output.resolve()
        if self.binary:
            return [self.binary, '--EmitONNXIR', '--mlir-print-op-generic', str(source), '-o', str(output / 'model')]
        if ',' in str(source.parent) or ',' in str(output):
            raise ValueError('Docker bind paths containing commas are not supported')
        return ['docker', 'run', '--rm', '--pull=never', '--network=none',
                '--platform=' + PIN['platform'],
                '--mount', f'type=bind,source={source.parent},target=/input,readonly',
                '--mount', f'type=bind,source={output},target=/output',
                PIN['image'], '--EmitONNXIR', '--mlir-print-op-generic',
                '/input/' + source.name, '-o', '/output/model']

    def import_model(self, source: Path, output: Path) -> dict:
        command = self.command(source, output)
        container_name = None
        if not self.binary:
            container_name = 'npu-onnx-import-' + uuid.uuid4().hex
            command[3:3] = ['--name', container_name]
        log = output / 'compiler.log'
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace', timeout=self.timeout)
        except FileNotFoundError as error:
            log.write_text(str(error), encoding='utf-8')
            raise OnnxMlirError(f'Compiler or Docker executable not found; see {log}') from error
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or b''
            if isinstance(captured, bytes):
                captured = captured.decode('utf-8', errors='replace')
            log.write_text(captured, encoding='utf-8')
            if container_name:
                try:
                    cleanup = subprocess.run(['docker', 'rm', '--force', container_name],
                                             capture_output=True, text=True, timeout=30)
                    with log.open('a', encoding='utf-8') as stream:
                        stream.write(f'\nContainer cleanup (exit {cleanup.returncode}):\n')
                        stream.write(cleanup.stdout + cleanup.stderr)
                except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                    with log.open('a', encoding='utf-8') as stream:
                        stream.write(f'\nContainer cleanup failed: {cleanup_error}\n')
            raise OnnxMlirError(f'ONNX-MLIR import timed out; see {log}') from error
        except OSError as error:
            log.write_text(str(error), encoding='utf-8')
            raise OnnxMlirError(f'Cannot execute ONNX-MLIR; see {log}') from error
        log.write_text(process.stdout or '', encoding='utf-8')
        if process.returncode:
            raise OnnxMlirError(f'ONNX-MLIR import failed (exit {process.returncode}); see {log}')
        module = output / 'model.onnx.mlir'
        if not module.is_file() or module.stat().st_size == 0:
            raise OnnxMlirError(f'Compiler returned no nonempty ONNX MLIR module; see {log}')
        digest = hashlib.sha256()
        with module.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        return {'command': command, 'artifact': module.name, 'artifact_sha256': digest.hexdigest(), 'log': log.name,
                'toolchain': {'binary': self.binary, 'stage': 'import-only'} if self.binary else PIN,
                'provenance': 'unverified-native' if self.binary else 'pinned-image'}
