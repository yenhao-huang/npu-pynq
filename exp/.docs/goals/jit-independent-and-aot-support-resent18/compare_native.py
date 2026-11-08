"""Compare native compiled ResNet18 to the independent NumPy W8A8 logits."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def main():
    from src.runtime.core.image import ImageEngine,top_predictions
    model = ROOT/'examples/resnet18/model'
    started = time.perf_counter()
    engine = ImageEngine(model/'aot','sim')
    loaded = time.perf_counter()
    logits = engine.predict(np.load(model/'reference-input.npy',allow_pickle=False))
    reference = np.load(model/'reference-logits.npy',allow_pickle=False)
    labels = (model/'imagenet-classes.txt').read_text().splitlines()
    top5 = top_predictions(logits,labels)
    record = {'backend':'native-host-simulated-npu','load_seconds':loaded-started,
              'inference_seconds':engine.last_seconds,'npu':engine.last_stats,
              'max_abs_logit_difference':float(np.abs(logits-reference).max()),
              'top5':[vars(p) for p in top5]}
    (model/'native-host.json').write_text(json.dumps(record,indent=2)+'\n')
    np.save(model/'native-host-logits.npy',logits)
    print(json.dumps(record,indent=2),flush=True)
    np.testing.assert_allclose(logits,reference,rtol=2e-3,atol=2e-3)
    if top5[0].name != 'Samoyed': raise RuntimeError('Native image classification failed')


if __name__ == '__main__': main()
