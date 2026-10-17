"""Verify tokenizer IDs and exact official Jinja template without board dependencies."""
import json
from pathlib import Path
import sys
from jinja2 import Environment
from tokenizers import Tokenizer

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parents[2]))
from official_tokenizer import Tokenizer

package=root/'packages/onnx'
actual=Tokenizer(package)
reference=Tokenizer.from_file(str(package/'tokenizer.json'))
template=Environment().from_string(json.loads((package/'tokenizer_config.json').read_text())['chat_template'])
cases=[[],[{'role':'user','content':'write a paragraph about npu'}],
       [{'role':'system','content':'Answer briefly.'},{'role':'user','content':'What is an NPU?'}],
       [{'role':'user','content':'Explain why the sky is blue in simple words.'}],
       [{'role':'user','content':'繁體中文、1234、hello world!'}]]
for messages in cases:
    expected=template.render(messages=messages,add_generation_prompt=True)
    value=actual.chat(messages)
    assert expected==value,(expected,value)
    assert reference.encode(expected).ids==actual.encode(value)
result={'pass':True,'cases':len(cases),'default_system_template':True}
(root/'runs/official-reference/tokenizer-check.json').write_text(json.dumps(result,indent=2))
print(result)
