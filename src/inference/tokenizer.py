"""Byte-level BPE tokenizer driven by a Hugging Face ``tokenizer.json``.

Implements what SmolLM and Qwen use: an optional NFC normalizer, a regex
pre-tokenizer (GPT-2's or a Split pattern, optionally preceded by Digits),
the GPT-2 byte-to-unicode mapping, rank-ordered merges, and added (special)
tokens matched verbatim. ``\\p{L}``/``\\p{N}``/``\\p{M}`` come from the
``regex`` package when present, otherwise from unicodedata-built classes so
the board needs no extra packages.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import unicodedata

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache(maxsize=None)
def _category_class(prefix: str) -> str:
    """A ``re`` character-class body for every code point in a Unicode category."""
    ranges = []
    start = prev = None
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp)).startswith(prefix):
            if start is None:
                start = prev = cp
            elif cp == prev + 1:
                prev = cp
            else:
                ranges.append((start, prev))
                start = prev = cp
    if start is not None:
        ranges.append((start, prev))

    def esc(cp: int) -> str:
        return f"\\U{cp:08x}"

    return "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in ranges)


def compile_pattern(pattern: str):
    try:
        import regex
        return regex.compile(pattern)
    except ImportError:
        pass
    out = pattern
    for name in ("L", "N", "M"):
        cls = _category_class(name)
        # Inside a class ([...\p{L}...]) the body is spliced in; outside it is wrapped.
        out = re.sub(r"(\[[^\]]*?)\\p\{" + name + r"\}", lambda m: m.group(1) + cls, out)
        out = out.replace("\\p{" + name + "}", "[" + cls + "]")
    return re.compile(out)


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        spec = json.loads((path / "tokenizer.json" if path.is_dir() else path).read_text())
        model = spec["model"]
        if model["type"] != "BPE":
            raise ValueError("only BPE tokenizers are supported")
        self.vocab: dict[str, int] = model["vocab"]
        merges = model["merges"]
        pairs = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(pairs)}
        self.added = {t["content"]: t["id"] for t in spec.get("added_tokens", [])}
        self.special_ids = {t["id"] for t in spec.get("added_tokens", []) if t.get("special")}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.id_to_token.update({i: t for t, i in self.added.items()})
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        norm = spec.get("normalizer") or {}
        self.nfc = norm.get("type") == "NFC"
        self.digits = False
        pattern = None
        for pre in self._pretokenizers(spec.get("pre_tokenizer")):
            kind = pre["type"]
            if kind == "Digits":
                self.digits = pre.get("individual_digits", False)
            elif kind == "Split":
                pattern = pre["pattern"]["Regex"]
            elif kind == "ByteLevel" and pre.get("use_regex", True) and pattern is None:
                pattern = GPT2_PATTERN
        self.pattern = compile_pattern(pattern or GPT2_PATTERN)
        added = sorted(self.added, key=len, reverse=True)
        self.added_re = re.compile("|".join(re.escape(a) for a in added)) if added else None

    @staticmethod
    def _pretokenizers(pre):
        if not pre:
            return []
        if pre["type"] == "Sequence":
            return pre["pretokenizers"]
        return [pre]

    @lru_cache(maxsize=65536)
    def _bpe(self, word: str) -> tuple[int, ...]:
        parts = list(word)
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                rank = self.ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, i)
            if best is None:
                break
            i = best[1]
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return tuple(self.vocab[p] for p in parts)

    def _encode_text(self, text: str) -> list[int]:
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        pieces = [text]
        if self.digits:
            pieces = [p for chunk in pieces for p in re.split(r"(\d)", chunk) if p]
        ids: list[int] = []
        for piece in pieces:
            for match in self.pattern.findall(piece):
                if not match:
                    continue
                word = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
                ids.extend(self._bpe(word))
        return ids

    def encode(self, text: str) -> list[int]:
        if self.added_re is None:
            return self._encode_text(text)
        ids: list[int] = []
        last = 0
        for m in self.added_re.finditer(text):
            ids.extend(self._encode_text(text[last:m.start()]))
            ids.append(self.added[m.group(0)])
            last = m.end()
        ids.extend(self._encode_text(text[last:]))
        return ids

    def decode_bytes(self, ids: list[int], skip_special: bool = True) -> bytes:
        out = bytearray()
        for i in ids:
            token = self.id_to_token.get(int(i), "")
            if int(i) in self.added.values():
                if skip_special and int(i) in self.special_ids:
                    continue
                out += token.encode("utf-8")
                continue
            out += bytes(self.byte_decoder[c] for c in token)
        return bytes(out)

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.decode_bytes(ids, skip_special).decode("utf-8", errors="replace")

    def token_id(self, text: str) -> int:
        return self.added.get(text, self.vocab.get(text, -1))

    def chat(self, messages: list[dict[str, str]], add_generation_prompt: bool = True,
             think: bool | None = None) -> str:
        """ChatML, the template SmolLM2-Instruct and Qwen share."""
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if think is False:
                text += "<think>\n\n</think>\n\n"
        return text
