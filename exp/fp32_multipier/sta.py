"""OpenROAD post-mapping STA adapter, with explicit ASAP7 units and loads."""
from pathlib import Path
import re
import os
ROOT=Path(__file__).resolve().parent
PDK=Path(os.environ.get('EVAL_PDK',str(ROOT.parent.parent/'external/OpenROAD-flow-scripts/flow/platforms/asap7'))).resolve()
def measure(dest,run,image):
    rel=dest.relative_to(ROOT).as_posix()
    import json
    delay=json.loads((dest/'metrics.json').read_text())['target_delay_ps']
    script=f'''read_lef /pdk/lef/asap7_tech_1x_201209.lef
read_lef /pdk/lef/asap7sc7p5t_28_R_1x_220121a.lef
read_liberty /exp/build/pdk/asap7_rvt_tt.lib
read_verilog /exp/{rel}/mapped.v
link_design fp12_mul
create_clock -name virtual -period {delay}
set_input_delay 0 -clock virtual [all_inputs]
set_output_delay 0 -clock virtual [all_outputs]
set_input_transition 10 [all_inputs]
set_load 3.898 [all_outputs]
report_units
report_checks -path_delay max -format full_clock_expanded -digits 6
report_power -digits 9
exit
'''
    (dest/'sta.tcl').write_text(script)
    cmd=['docker','run','--rm','--network','none','--mount',f'type=bind,source={ROOT},target=/exp','--mount',f'type=bind,source={PDK},target=/pdk,readonly','--entrypoint','bash',image,'-lc',f'source /OpenROAD-flow-scripts/env.sh && openroad -exit /exp/{rel}/sta.tcl']
    run(cmd,dest/'sta.log')
    txt=(dest/'sta.log').read_text()
    if '[ERROR' in txt or re.search(r'^Error:',txt,re.M):raise RuntimeError('OpenROAD error in sta.log')
    match=re.search(r'^\s*([\d.]+)\s+data arrival time\s*$',txt,re.M)
    if not match:raise RuntimeError('Missing timing path arrival')
    delay=float(match.group(1))
    if delay<=0:raise RuntimeError('Invalid nonpositive delay')
    power=re.search(r'^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)',txt,re.M)
    result={'delay_ps':delay}
    if power:result['estimated_power_w']=float(power.group(4))
    return result
