"""Quantization quality against the FP32 model: next-token agreement, KL
divergence, and perplexity on a fixed evaluation text."""

from __future__ import annotations

import numpy as np

EVAL_TEXT = (
    "The history of computing is longer than the history of computing hardware and modern computing "
    "technology and includes the history of methods intended for pen and paper or for chalk and slate, "
    "with or without the aid of tables. Digital computing is intimately tied to the representation of "
    "numbers. But long before abstractions like the number arose, there were mathematical concepts to "
    "serve the purposes of civilization. These concepts are implicit in concrete practices such as "
    "one-to-one correspondence, the basis of counting, and comparison to a standard, used for measurement.\n"
    "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
    "        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
    "    return -1\n"
    "Photosynthesis is the process by which green plants and certain other organisms transform light "
    "energy into chemical energy. During photosynthesis in green plants, light energy is captured and "
    "used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds."
)


def compare_logits(ref: np.ndarray, test: np.ndarray, ids: list[int]) -> dict[str, float]:
    """ref/test: [T, V] logits for the same token ids; scores predict ids[1:]."""
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    def log_softmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    lr, lt = log_softmax(ref), log_softmax(test)
    kl = float(np.mean(np.sum(np.exp(lr) * (lr - lt), -1)))
    top1 = float(np.mean(ref.argmax(-1) == test.argmax(-1)))
    targets = np.asarray(ids[1:])
    nll_ref = -lr[np.arange(len(targets)), targets].mean()
    nll_test = -lt[np.arange(len(targets)), targets].mean()
    return {"top1_agreement": top1, "kl_divergence": kl,
            "ppl_fp32": float(np.exp(nll_ref)), "ppl_quant": float(np.exp(nll_test))}
