"""A timed-out EDA process must not leave its children alive."""
import subprocess
import sys
from pathlib import Path
from ic_core.process import run


def test_timeout_stops_owned_descendant_and_preserves_sibling(tmp_path):
    child=tmp_path/'child.py'
    child.write_text('from pathlib import Path\nimport time,signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nPath("started").write_text("yes")\ntime.sleep(2)\nPath("escaped").write_text("bad")\n')
    parent=tmp_path/'parent.py'
    parent.write_text('import subprocess,sys,time\nsubprocess.Popen([sys.executable,"child.py"])\ntime.sleep(30)\n')
    sibling_code='from pathlib import Path; import time; time.sleep(2.5); Path("sibling_survived").write_text("yes")'
    sibling=subprocess.Popen([sys.executable,'-c',sibling_code],cwd=tmp_path)
    try:
        result=run([sys.executable,str(parent)],cwd=tmp_path,log_path=tmp_path/'process.log',timeout_s=1)
        assert result.timed_out
        assert (tmp_path/'started').exists()
        assert sibling.wait(timeout=10)==0
        assert (tmp_path/'sibling_survived').exists()
        assert not (tmp_path/'escaped').exists()
    finally:
        if sibling.poll() is None:
            sibling.terminate();sibling.wait(timeout=5)
