import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.runtime.core.image import ImageEngine, preprocess, top_predictions


CONTRACT = {'shape':[224,224,3], 'layout':'NHWC',
            'mean':[0.485,0.456,0.406], 'std':[0.229,0.224,0.225]}


class FakePackage:
    def __init__(self, *args, **kwargs):
        self.manifest = {'meta':{'input':CONTRACT, 'classes':5},
                         'entry_points':{'forward':['weights','image','logits']}}
        self.count = 0

    def stats(self):
        return {k:self.count for k in ('calls','jobs','cycles','macs','polls')}

    def call(self, name, weights, image, logits):
        assert (name, weights) == ('forward','weights')
        self.count += 1
        logits[:] = [[0,1,4,2,-1]]


class ImageTests(unittest.TestCase):
    def test_preprocessing_preserves_channels_and_normalization(self):
        image = Image.new('RGB',(319,257),(255,128,0))
        result = preprocess(image, CONTRACT)
        expected = (np.array([1,128/255,0],np.float32)-CONTRACT['mean'])/CONTRACT['std']
        self.assertEqual(result.shape,(224,224,3))
        self.assertEqual(result.dtype,np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(result[0,0],expected,rtol=1e-6)

    @patch('src.runtime.core.image.Package',FakePackage)
    def test_abi_prediction_and_counter_deltas(self):
        engine = ImageEngine('unused','sim')
        image = np.zeros((224,224,3),np.float32)
        first = engine.predict(image)
        second = engine.predict(image[:,::-1])
        np.testing.assert_array_equal(first,second)
        self.assertEqual(engine.last_stats['jobs'],1)
        top = top_predictions(second,['a','b','c','d','e'])
        self.assertEqual(top[0].name,'c')
        self.assertAlmostEqual(sum(p.probability for p in top),1)

    @patch('src.runtime.core.image.Package',FakePackage)
    def test_reject_wrong_input_before_native_call(self):
        engine = ImageEngine('unused','sim')
        for value in (np.zeros((3,224,224),np.float32), np.zeros((224,224,3),np.int8),
                      np.full((224,224,3),np.nan,np.float32)):
            with self.assertRaises(ValueError): engine.predict(value)
        self.assertEqual(engine.package.count,0)

    def test_notebook_keeps_six_stages_and_upload_ui(self):
        notebook = json.loads((Path(__file__).resolve().parents[1]/'resnet18.ipynb').read_text(encoding='utf-8'))
        code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),6)
        self.assertIn('widgets.FileUpload',code[2])
        self.assertIn('uploaded_file',code[3])
        self.assertIn('engine.predict',code[4])
        self.assertIn('top_predictions',code[5])
        for cell in code: compile(cell,'<notebook>','exec')
        self.assertNotIn('src.jit','\n'.join(code))


if __name__ == '__main__': unittest.main()
