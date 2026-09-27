# Clocked physical measurement

Operation: `clocked_ppa`. This extends ASPEN-style physical feedback with
matched registered boundaries, single-clock OOC implementation, LUT/FF/DSP/BRAM
counts, and estimated throughput normalized by a caller-verified initiation
interval. It does not reproduce ASPEN's search algorithm.

Run the shared substantive reduction study:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-14-rover/study.py
```

The study requests a 5 ns clock on xc7z020clg400-1, with Default placement and
routing, one-cycle latency and II=1. Negative slack is retained, not called a
successful 200 MHz implementation. Estimated Fmax comes from period minus
worst register-to-register setup slack; it is not board validation. Every
resource class is reported, including zero DSP and BRAM counts.
