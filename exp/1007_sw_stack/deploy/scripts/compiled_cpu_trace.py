"""Run the identical ARM compiled library on ISA simulation in its own process."""
import argparse
import sys
from pathlib import Path

repo=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(repo))
from src.inference.engine import LLMEngine
from official_acceptance import trace
from acceptance import save

p=argparse.ArgumentParser()
p.add_argument('--package',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
engine=LLMEngine(a.package,backend='cpu')
s=engine.model.lib.npu_rt_stats().contents
assert s.backend==0 and not s.mmio
trace(engine,a.output,'cpu-reference')
save(Path(a.output)/'cpu-runtime.json',{'backend':s.backend,'mmio':bool(s.mmio),
    'compiled_pipeline':engine.meta['compiled_pipeline'],'stats':engine.model.stats()})
