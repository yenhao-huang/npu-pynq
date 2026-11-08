"""Independent NumPy execution of the frontend's W8A8 image contract."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from src.export.graphs.resnet18.model import folded_convs, preprocess
    from src.export.compilers.mlir.ops.conv import ref_im2col
    from src.export.compilers.mlir.ops.linear import quantize_weight, ref_linear
    from src.runtime.core.image import top_predictions
    model = ROOT/'examples/resnet18/model'
    state = {k:v.detach().numpy() for k,v in torch.load(model/'resnet18-f37072fd.pth',
              map_location='cpu',weights_only=True).items()}
    convs = folded_convs(state)
    if (model/'reference-input.npy').is_file():
        x = np.load(model/'reference-input.npy',allow_pickle=False)
    else:
        from PIL import Image
        with Image.open(model/'demo-image.jpg') as image:
            x = preprocess(image)
    times = {}

    def conv(x,h,w,c,name,k,stride,pad,relu):
        weight,bias = convs[name]
        cols = ref_im2col(x,h,w,c,k,stride,pad)
        y = ref_linear(cols,quantize_weight(weight,8),bias)
        oh,ow = (h+2*pad-k)//stride+1,(w+2*pad-k)//stride+1
        return (np.maximum(y,0) if relu else y),oh,ow

    start = time.perf_counter()
    t = start
    x,h,w = conv(x.reshape(-1,3),224,224,3,'conv1',7,2,3,True)
    padded = np.pad(x.reshape(h,w,64),((1,1),(1,1),(0,0)),constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded,(3,3),axis=(0,1))
    x = windows[::2,::2].max(axis=(-1,-2)).reshape(-1,64)
    h,w,c = 56,56,64
    times['stem'] = time.perf_counter()-t
    for stage,out_c in enumerate((64,128,256,512),start=1):
        t = time.perf_counter()
        for block in range(2):
            prefix = f'layer{stage}.{block}.'
            stride = 2 if block==0 and stage>1 else 1
            y,oh,ow = conv(x,h,w,c,prefix+'conv1',3,stride,1,True)
            y,oh,ow = conv(y,oh,ow,out_c,prefix+'conv2',3,1,1,False)
            skip = conv(x,h,w,c,prefix+'downsample.0',1,stride,0,False)[0] if prefix+'downsample.0' in convs else x
            x = np.maximum(y+skip,0)
            h,w,c = oh,ow,out_c
        times[f'layer{stage}'] = time.perf_counter()-t
    t = time.perf_counter()
    pooled = (x.sum(axis=0,keepdims=True)*np.float32(1/x.shape[0])).astype(np.float32)
    logits = ref_linear(pooled,quantize_weight(state['fc.weight'],8),state['fc.bias'])[0]
    times['classifier'] = time.perf_counter()-t
    labels = (model/'imagenet-classes.txt').read_text().splitlines()
    top5 = top_predictions(logits,labels)
    np.save(model/'reference-logits.npy',logits)
    record = {'backend':'numpy-W8A8-reference','block_seconds':times,
              'total_seconds':time.perf_counter()-start,'top5':[vars(p) for p in top5]}
    (model/'reference-image.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    if top5[0].name != 'Samoyed': raise RuntimeError('Reference prediction mismatch')


if __name__ == '__main__': main()

