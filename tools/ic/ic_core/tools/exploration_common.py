"""Shared source identity and conservative identifier handling for exploration."""
import hashlib
import re
from pathlib import Path

from ..errors import InvalidInput
from pydantic import Field
from typing import Annotated

Identifier = Annotated[str, Field(pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')]


def sources(files, cwd):
    paths = [(Path(cwd) / name).resolve() for name in files]
    if not paths or any(not p.is_file() for p in paths):
        raise InvalidInput('every source must be an existing file')
    return paths


def fingerprint(files, top):
    digest = hashlib.sha256(top.encode())
    for path in files:
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()


def tcl_path(path):
    text = Path(path).as_posix()
    if any(c in text for c in '{}\n\r'):
        raise InvalidInput('Tcl paths must not contain braces or newlines')
    return '{' + text + '}'


def require_combinational(files):
    # Conservative boundary, not an HDL parser or a proof of combinationality.
    for path in files:
        text = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(encoding='utf-8'), flags=re.S)
        if re.search(r'\b(posedge|negedge|initial|always_ff|always_latch)\b|#\s*\d', text):
            raise InvalidInput('sequential or timed RTL is outside the combinational checker contract')

# argparse transports complex schema fields as JSON strings. Keep decoding in
# the operation contract so every generated client uses the same validation.
import json
from pydantic import model_validator
from .common import Backendable


class ExplorationInput(Backendable):
    @model_validator(mode='before')
    @classmethod
    def decode_json_fields(cls, value):
        if not isinstance(value, dict):
            return value
        decoded = dict(value)
        if decoded.get("backend") is None:
            decoded["backend"] = cls.model_fields["backend"].default
        for key, item in decoded.items():
            if isinstance(item, str) and item.lstrip().startswith(('{', '[')):
                decoded[key] = json.loads(item)
            elif isinstance(item, list):
                decoded[key] = [json.loads(v) if isinstance(v, str) and v.lstrip().startswith('{') else v for v in item]
        return decoded
