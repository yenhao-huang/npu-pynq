"""Execute the six original notebook stages on PYNQ and retain board evidence."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import nbformat
from nbclient import NotebookClient


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0,str(root))
    from package_example import verify_delivery
    verify_delivery(root)
    notebook = nbformat.read(root/'resnet18.ipynb',as_version=4)
    started = time.perf_counter()
    try:
        NotebookClient(notebook, timeout=3600, kernel_name='python3',
                       resources={'metadata':{'path':str(root)}},
                       store_widget_state=False).execute()
    finally:
        nbformat.write(notebook, root/'resnet18.executed.ipynb')
    text = ''.join(out.get('text','') for cell in notebook.cells[-1:]
                   for out in cell.get('outputs',[]) if out.output_type=='stream')
    offset = text.index('{\n  "block_seconds"')
    result = json.loads(text[offset:])
    result.update(hostname=platform.node(), machine=platform.machine(), python=sys.version,
                  notebook_wall_seconds=time.perf_counter()-started,
                  package_manifest_sha256=sha(root/'model/aot/manifest.json'),
                  overlay_sha256=sha(root/'resnet18_aot.bit'),
                  hwh_sha256=sha(root/'resnet18_aot.hwh'),
                  image_sha256=sha(root/'model/demo-image.jpg'))
    (root/'notebook-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['backend'] != 'pynq' or result['npu']['jobs'] <= 0 or not result['correct']:
        raise RuntimeError('Physical notebook acceptance failed')


if __name__ == '__main__': main()
