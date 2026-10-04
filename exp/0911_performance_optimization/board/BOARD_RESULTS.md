# Three-design physical ResNet-18 benchmark

All three designs executed all 21 notebook cells (10 code cells) on the same PYNQ-Z1. Seven matrix smoke cases per design and every pinned output digest passed. CSV job counts, MACs, cycles and DMA bytes independently match the section-7 aggregates.

## 7. Execute ResNet-18 on the NPU

| Design | Full inference (s) | CPU (s) | Speedup vs 09 | Images/s | Physical jobs | Physical cycles | Busy cycles / clock (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 09 | 1905.466489 | 1892.181990 | 1.0000x | 0.00052481 | 35174 | 4428501001 | 44.285010 |
| 19 | 1892.457117 | 1879.513679 | 1.0069x | 0.00052841 | 35174 | 4423270392 | 44.232704 |
| 21 | 1892.007265 | 1879.394515 | 1.0071x | 0.00052854 | 35174 | 4424804404 | 44.248044 |

Each forward performs 1,814,073,344 MACs at a read-back FCLK0 of 100 MHz. One measured full forward per design; observed differences are not a statistical confidence interval. Design 09 is the comparison reference among these three optimized 16x16 designs.

The timing boundary includes constructing NPUModelRuntime, complete model execution, CPU lowering/software operators, physical jobs and trace/checkpoint overhead. It excludes model loading, FPGA programming, matrix smoke, output digest checks and report writing. Physical cycles include producer/DMA stalls; cycles divided by FCLK are not pure MAC compute time. PYNQ Clocks reports the register-derived frequency, not an external instrument measurement.

## Interpretation

All three measured full-forward times are within 0.71%. The small single-sample difference does not establish a hardware speed ranking. Approximately 84% of wall time lies outside physical job calls, and process CPU time is close to wall time. The main end-to-end limitation is therefore in the host software path. The phase counters do not identify individual hot functions; CPU lowering and software operators are the next profiling targets.

| Host / watchdog observation | 09 | 19 | 21 |
| --- | ---: | ---: | ---: |
| Wall time outside physical jobs (%) | 84.033 | 84.603 | 84.443 |
| Maximum single-job cycles | 994739 | 1193343 | 989749 |
| Correct jobs exceeding old 10 ms limit | 0 | 1 | 0 |

Design 19 contains one correctly completed 11.93343 ms job. The old 10 ms watchdog would reject such a valid job. Designs 09 and 21 also approach that old limit. The 1 second watchdog preserves a finite failure bound and all results still require full numeric validation.

The 09-versus-19 wall-time difference is 13.009372 seconds; allocation/padding/copy accounts for 13.004274 seconds of that difference. Busy cycles / clock differ by only 0.052306 seconds. This does not prove that the allocation difference was caused by hardware. For resource trade-offs, 19 uses only 12 BRAM tiles and has the largest positive setup slack; 21 uses all 220 DSPs without a decisive measured full-forward latency advantage.

## Physical submission phases

| Phase (seconds) | 09 | 19 | 21 |
| --- | ---: | ---: | ---: |
| allocate_pad_copy_seconds | 173.889825 | 160.885551 | 164.124119 |
| configure_start_seconds | 15.122930 | 15.188842 | 15.241410 |
| a_flush_dma_wait_seconds | 25.452710 | 25.465196 | 25.404798 |
| b_flush_dma_wait_seconds | 22.938298 | 22.998361 | 23.059628 |
| completion_receive_wait_seconds | 16.050065 | 15.987028 | 16.051828 |
| invalidate_unpad_copy_seconds | 8.584727 | 8.675557 | 8.693233 |
| free_buffers_seconds | 36.753479 | 36.676151 | 36.264484 |
| job_wall_seconds | 304.253920 | 291.373520 | 294.336793 |

DMA phase timers include host driver, cache maintenance and polling. Phase times are nested within physical job wall time and must not be added to it. The remainder of full inference includes CPU model work and instrumentation outside physical submissions.

| Data / memory | 09 | 19 | 21 |
| --- | ---: | ---: | ---: |
| logical_input_bytes | 236676608 | 236676608 | 236676608 |
| logical_output_bytes | 33525568 | 33525568 | 33525568 |
| a_dma_bytes | 113630720 | 113630720 | 113630720 |
| b_dma_bytes | 123296768 | 123296768 | 123296768 |
| c_dma_bytes | 33525568 | 33525568 | 33525568 |
| Peak RSS (KiB) | 220632 | 227536 | 220692 |

## Complete-overlay implementation

| Design | LUT | FF | BRAM tiles | DSP | WNS (ns) | WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 09 | 32683 | 21285 | 74.5 | 192 | +0.105 | +0.042 |
| 19 | 37662 | 24548 | 12 | 192 | +0.186 | +0.031 |
| 21 | 35670 | 23545 | 76 | 220 | +0.016 | +0.052 |

All complete PS7/DMA/accelerator overlays pass 100 MHz setup/hold/pulse, routing, DRC and timing-coverage gates. Designs 09 and 21 use post-route physical refinement with unchanged RTL.

## Notebook code-cell wall times

| Cell / step | 09 (s) | 19 (s) | 21 (s) |
| --- | ---: | ---: | ---: |
| 3 / step 1 | 1.412423 | 1.025773 | 0.933046 |
| 5 / step 2 | 0.530450 | 0.575402 | 0.763060 |
| 7 / step 3 | 7.334548 | 6.202660 | 7.323730 |
| 9 / step 4 | 0.461895 | 0.460225 | 0.456393 |
| 11 / step 5 | 162.534850 | 160.389250 | 162.617500 |
| 13 / step 6 | 6.976006 | 5.366394 | 5.872796 |
| 15 / step 7 | 1905.611339 | 1892.604172 | 1892.146656 |
| 17 / step 8 | 0.170546 | 0.180890 | 0.171767 |
| 19 / step 9 | 0.246418 | 0.256748 | 0.251174 |
| 21 / step 10 | 0.108716 | 0.122979 | 0.123811 |

Notebook-cell timing includes Jupyter execution overhead; the section-7 timer inside cell 15 is the primary inference measurement.

## Integration finding and retained evidence

The first design-19 attempt stopped before inference: HWH divisors 4 x 4 produced 62.5 MHz with the board boot PLL configuration. The shared runtime now requests and verifies 100 MHz after overlay download. All three use identical corrected runtime/notebook code; BIT/HWH are unchanged. First-attempt evidence and original deployment packages are preserved.

Each board_expXX/artifacts/board_result contains resnet18.executed.ipynb, runner.log and evidence/{cells.json,matrix_smoke.json,section7.json,physical_jobs.csv,board_benchmark.json}. Raw reports and hash manifests remain in artifacts/deployment. Host-timestamped monitoring samples are in board/scratch/board_poll_history.jsonl.

Executed notebooks: [09](../board_exp09/artifacts/board_result/resnet18.executed.ipynb), [19](../board_exp19/artifacts/board_result/resnet18.executed.ipynb), [21](../board_exp21/artifacts/board_result/resnet18.executed.ipynb).

A second initial design-19 run stopped at 93.4% MAC progress when its 10 ms watchdog expired awaiting B input. That tile shape had previously succeeded 896 times. A controlled 20 ms producer delay reproduced the timeout, while a 1 second watchdog passed that delay and 1,000 additional tail/full matrix tests. Final runs all use a 1 second hardware watchdog and 5 second per-job software limit. Failed partial runs are preserved and excluded from full-forward comparisons; the exact OS cause of the original delay was not instrumented.

Board calendar time is incorrect (2025); host collection labels use 2026-09-11 UTC. All elapsed measurements use monotonic clocks. Evidence is automated development validation, not human-reviewed acceptance.
