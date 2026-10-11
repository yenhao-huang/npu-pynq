# A/B 實驗報告：IC 工具是否提升 PPA

依 `docs/goals/ab_test_human.md` 執行。所有數字由 `analysis/analyze.py` 從原始紀錄重新計算；重現方式見 [Reproduce.md](Reproduce.md)。原始逐次紀錄與 RTL 快照保留在本機封存，未收入此精簡版 PR；單靠這份 Git checkout 無法獨立重新計算歷史數字。

2026-10-04 更新：附錄 B 加入「baseline + shell」（不給 IC 工具，但可以用 shell），回答「模型自己會用 iverilog / yosys 時，IC 工具還有沒有幫助」。

## 結論

1. **正確性沒有因工具而提升。** single-round 加工具後，pass@1 的變化是 Opus +1.3、Sonnet +0.7、gpt-6-sol −1.4 個百分點，都在雜訊範圍內（每組 150 次 attempt，標準誤約 3–4 個百分點）。multi-round 以最後一輪計算的 pass@1 反而略低（−0.7 / −8.0 / −4.7），因為後續輪次為了探索 PPA 偶爾會交出錯誤設計。語法合法率在所有組都 ≥ 99%。
2. **PPA：single-round 加工具有小幅改善，前提是模型真的有用工具。** 同一題最佳 PDAP 相對 baseline 的幾何平均：Opus 0.879×（勝 / 平 / 負 = 12 / 22 / 4）、gpt-6-sol 0.870×（13 / 12 / 2）。Sonnet 是 1.014×（2 / 24 / 5），因為 150 次 attempt 裡只有 7 次呼叫了 IC 工具，而且都只用 lint。
3. **PPA：multi-round 改善最明顯。** PDAP 幾何平均為 Opus 0.587×、Sonnet 0.746×、gpt-6-sol 0.709×，勝場分別是 34 / 38、20 / 27、24 / 25 題。把 multi-round 限制在前 3 輪，讓候選數和 baseline 一樣是 3 個，仍有 0.676× / 0.779× / 0.809×。只看第 1 輪則和 baseline 相當（0.977× / 0.995× / 0.995×）。可見改善來自多輪探索，第一個答案並沒有比較好。
4. **三組合併比較時，frontier 上的點主要來自 multi-round。** 以「在該題持有最低 PDAP」計算的題數，baseline / single / multi 分別是：Opus 5 / 12 / 34、Sonnet 10 / 13 / 26、gpt-6-sol 3 / 5 / 24。
5. **代價高。** Opus multi-round 花了 $63.07（50 次 attempt），是 baseline 的 5.2 倍（$12.14，150 次）、single-round 的 2.9 倍（$21.57）。gpt-6-sol multi-round 用掉約 1.2 億 cached input tokens，是 single-round 的 3.8 倍（3,070 萬）。
6. **無法區分「工具」和「多輪」各自的貢獻。** 依目標文件只跑了三組，沒有「多輪但不給工具」的對照組，所以 multi-round 的改善不能全部歸功於工具（見「限制」）。
7. **只給 shell 不會改善 PPA（附錄 B）。** 不給 IC 工具、只給 shell 時，同題最佳 PDAP 相對 baseline 為 Opus 1.011×、Sonnet 0.996×、gpt-6-sol 1.062×；IC 工具組（尤其 multi-round）仍明顯較好。

## 實驗設定

| 項目 | 設定 |
|---|---|
| 模型 | `claude-opus-5-5`、`claude-sonnet-5-5`（Claude Code CLI 2.1.288）、`gpt-6-sol`（Codex CLI 0.160.0） |
| effort | `medium`（Claude `--effort medium`；Codex `model_reasoning_effort="medium"`） |
| 題目 | RTLLM v2.0（pin `9d5f61a`）全部 50 題 |
| 重複 | baseline / single-round 每題獨立 3 次；multi-round 每題 1 次（依使用者指示，為了節省 token） |
| 輪數 | baseline / single-round 1 輪；multi-round 5 輪（目標文件表格範本寫 3，文字寫「暫設 5」，採 5） |
| 每輪預算 | Claude `$3.00` / 次呼叫；Codex 沒有預算參數，無上限 |
| 工具 | `tools` 組開放 `lint`、`sim`、`signals`、`first_mismatch`、`value_at`、`value_range`、`synth`、`ppa`、`equiv`（工具庫 `npu/issue74-b` @ `cdb514f`，不含 `show_wave`）；檔案操作只能用 Write/Read/Edit（Codex 為 `apply_patch`），不能用 shell |
| PPA 量測 | Nangate45、時脈週期 10 ns、input activity 0.1、OpenROAD 26Q1-1744；`delay_ns = 1000 / fmax_mhz` |
| 正確性 | RTLLM 隨附 testbench 加上 held-out 隨機差分測試（`radix2_div` 依既有文件略過 held-out） |

### 指標定義

- **Success rate**：語法合法率，即提交的 snapshot 中，單獨用 `iverilog -g2012` 能編譯通過的比例。沒有交出檔案視為不合法。multi-round 以每一輪的 snapshot 為分母。
- **pass@1**：每次 attempt 最後交付的 snapshot（multi-round 為第 5 輪或中止前最後一輪）通過正確性檢查的比例，先在題內平均，再對題目平均。
- **pass@3**：3 次獨立 attempt 中至少一次最後交付正確的題目比例。multi-round 只有 1 次 attempt，因此為 N/A。
- **每題 Pareto 候選數**：該組在同一題的所有有效候選（正確、`ppa_ok`、三個指標皆為有限正數）形成的 frontier 點數，對 50 題取平均（沒有有效候選的題目記為 0）。完全相同的點都保留。
- **Frontier Top 3：PDAP**：先取各組在該題的 frontier，再依 `PDAP = power_w × delay_ns × area_um2` 由小到大取前 3。PDAP 只用來排序，不影響 frontier 的判定，也不跨題比較。
- **遠強於 baseline 題數**：該組在同一題存在一個有效候選，其 area、power、delay 三項都不差於 baseline 的最佳候選，且至少一項嚴格較好（相對容差 1e-6）。這樣的題目數。只有面積換功耗這類取捨不計入。
- **PDAP / baseline**：每題若該組有「遠強於 baseline」的候選，取其中 PDAP 最低者除以 baseline 最佳 PDAP；沒有則記為 1.000×，代表該題維持 baseline。再對兩組都有有效候選的題目取幾何平均。數值越低越好。不設此限制的版本見「補充比較」。
- **Power、Area、Delay**：沿用 PDAP / baseline 的逐題配對及選出的候選，分別對原始 `power_w`、`area_um2`、`delay_ns` 取算術平均。Power 換算為 µW；沒有遠強於 baseline 的候選時，採用同題 baseline 最佳 PDAP 候選的原始數值。各列的有效題目集合可能不同，原始平均值不宜直接當作配對改善幅度。

## 比較表格

### Opus 5.5

**實驗設定**：模型 `claude-opus-5-5`｜effort `medium`｜題數 `50`｜每題獨立重複：baseline / single-round `3` 次，multi-round `1` 次｜每輪預算 $3.00 / 次呼叫

| 實驗組 | Round | Success rate | pass@3 | pass@1 | 遠強於 baseline 題數 | Power (µW) | Area (µm²) | Delay (ns) | PDAP / baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (`notools`) | 1 | 100.0% | 82.0% | 78.7% | 0 | 204.12 | 389.69 | 0.98 | 1.000× (n=39) |
| single-round (`tools`) | 1 | 100.0% | 84.0% | 80.0% | 9 | 198.82 | 326.73 | 0.93 | 0.894× (n=38) |
| multi-round (`tools`) | 5 | 100.0% | N/A | 78.0% | 19 | 196.08 | 320.75 | 0.92 | 0.699× (n=38) |

### Sonnet 5.5

**實驗設定**：模型 `claude-sonnet-5-5`｜effort `medium`｜題數 `50`｜每題獨立重複：baseline / single-round `3` 次，multi-round `1` 次｜每輪預算 $3.00 / 次呼叫

| 實驗組 | Round | Success rate | pass@3 | pass@1 | 遠強於 baseline 題數 | Power (µW) | Area (µm²) | Delay (ns) | PDAP / baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (`notools`) | 1 | 99.3% | 66.0% | 64.0% | 0 | 219.19 | 327.90 | 0.69 | 1.000× (n=31) |
| single-round (`tools`) | 1 | 100.0% | 72.0% | 64.7% | 2 | 219.18 | 327.71 | 0.68 | 0.995× (n=31) |
| multi-round (`tools`) | 5 | 99.6% | N/A | 56.0% | 12 | 235.84 | 297.07 | 0.69 | 0.808× (n=27) |

### gpt-6-sol

**實驗設定**：模型 `gpt-6-sol`｜effort `medium`｜題數 `50`｜每題獨立重複：baseline / single-round `3` 次，multi-round `1` 次｜每輪預算 無（Codex CLI 沒有預算參數）

| 實驗組 | Round | Success rate | pass@3 | pass@1 | 遠強於 baseline 題數 | Power (µW) | Area (µm²) | Delay (ns) | PDAP / baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline (`notools`) | 1 | 100.0% | 62.0% | 58.7% | 0 | 137.25 | 291.14 | 0.51 | 1.000× (n=28) |
| single-round (`tools`) | 1 | 99.3% | 60.0% | 57.3% | 7 | 138.76 | 280.28 | 0.47 | 0.919× (n=27) |
| multi-round (`tools`) | 5 | 99.2% | N/A | 54.0% | 20 | 143.80 | 265.94 | 0.46 | 0.793× (n=25) |

Success rate、pass@3、pass@1、遠強於 baseline 題數、三個原始 PPA 平均值及 PDAP / baseline 的定義見「指標定義」。n 是兩組都有有效候選的題數，也是三個原始平均值與 PDAP 比值的分母；題數欄是其中至少一項改善、其餘不差的題數。各組逐題的 Frontier Top 3 PDAP 見附錄 A；baseline + shell 的數據見附錄 B。

表中的 PPA 候選只採用三項都不差、至少一項改善的設計，不採用面積換功耗這類取捨，所以 PDAP 比值比不設限的版本保守。不設限時 PDAP 比值為 Opus 0.879× / 0.587×、Sonnet 1.014× / 0.746×、gpt-6-sol 0.870× / 0.709×（single-round / multi-round），逐題勝負見「補充比較」。

## 補充比較

### 正確性

| 模型 | 實驗組 | Attempts | pass@1 | pass@3 | 任一輪正確 | FUNCTION |
|---|---|---:|---:|---:|---:|---:|
| Opus 5.5 | baseline | 150 | 78.7% | 82.0% | 78.7% | 41 / 50 |
| Opus 5.5 | single-round | 150 | 80.0% | 84.0% | 80.0% | 42 / 50 |
| Opus 5.5 | multi-round | 50 | 78.0% | N/A | 80.0% | 40 / 50 |
| Sonnet 5.5 | baseline | 150 | 64.0% | 66.0% | 64.0% | 33 / 50 |
| Sonnet 5.5 | single-round | 150 | 64.7% | 72.0% | 64.7% | 36 / 50 |
| Sonnet 5.5 | multi-round | 50 | 56.0% | N/A | 58.0% | 29 / 50 |
| gpt-6-sol | baseline | 150 | 58.7% | 62.0% | 58.7% | 31 / 50 |
| gpt-6-sol | single-round | 150 | 57.3% | 60.0% | 57.3% | 30 / 50 |
| gpt-6-sol | multi-round | 50 | 54.0% | N/A | 56.0% | 28 / 50 |

FUNCTION 的分母相同，但 baseline / single-round 每題有 3 次 attempt，multi-round 只有 1 次（5 輪），兩者抽樣次數不同。

### PPA：同題最佳 PDAP 相對 baseline

勝 / 平 / 負以同一題的最佳 PDAP 比較（相對誤差 1e-6 內算平手）；n 為兩組都有有效候選的題數。

| 模型 | 比較 | PDAP / baseline（幾何平均） | n | 勝 / 平 / 負 | 平均候選數 |
|---|---|---:|---:|---|---:|
| Opus 5.5 | single-round | 0.879× | 38 | 12 / 22 / 4 | 1.86 |
| Opus 5.5 | multi-round（5 輪） | 0.587× | 38 | 34 / 2 / 2 | 2.98 |
| Opus 5.5 | multi-round 前 3 輪 | 0.676× | 38 | 30 / 6 / 2 | 1.96 |
| Opus 5.5 | multi-round 第 1 輪 | 0.977× | 38 | 10 / 19 / 9 | 0.76 |
| Sonnet 5.5 | single-round | 1.014× | 31 | 2 / 24 / 5 | 1.78 |
| Sonnet 5.5 | multi-round（5 輪） | 0.746× | 27 | 20 / 6 / 1 | 2.14 |
| Sonnet 5.5 | multi-round 前 3 輪 | 0.779× | 26 | 18 / 8 / 0 | 1.32 |
| Sonnet 5.5 | multi-round 第 1 輪 | 0.995× | 26 | 2 / 20 / 4 | 0.52 |
| gpt-6-sol | single-round | 0.870× | 27 | 13 / 12 / 2 | 1.22 |
| gpt-6-sol | multi-round（5 輪） | 0.709× | 25 | 24 / 0 / 1 | 1.76 |
| gpt-6-sol | multi-round 前 3 輪 | 0.809× | 25 | 21 / 2 / 2 | 1.22 |
| gpt-6-sol | multi-round 第 1 輪 | 0.995× | 25 | 7 / 13 / 5 | 0.50 |

「前 3 輪」讓 multi-round 和 baseline 一樣只有 3 個候選，這是比較公平的比較。「第 1 輪」只有 1 個候選，baseline 有 3 個，因此對 multi-round 不利。

### 三組合併的 frontier

每題把三組所有有效候選放在一起取 frontier。

| 模型 | 實驗組 | 貢獻的 frontier 點數 | 進入合併 frontier 的題數 | 持有最低 PDAP 的題數（平手各計） |
|---|---|---:|---:|---:|
| Opus 5.5 | baseline | 42 | 18 | 5 |
| Opus 5.5 | single-round | 59 | 27 | 12 |
| Opus 5.5 | multi-round | 143 | 39 | 34 |
| Sonnet 5.5 | baseline | 45 | 18 | 10 |
| Sonnet 5.5 | single-round | 49 | 19 | 13 |
| Sonnet 5.5 | multi-round | 107 | 27 | 26 |
| gpt-6-sol | baseline | 14 | 7 | 3 |
| gpt-6-sol | single-round | 28 | 15 | 5 |
| gpt-6-sol | multi-round | 87 | 26 | 24 |

### 工具使用與成本

| 模型 | 實驗組 | 呼叫 IC 工具的 attempt | IC 工具呼叫次數 | 成本 / token | 模型耗時 |
|---|---|---:|---|---|---:|
| Opus 5.5 | baseline | 0 / 150 | — | $12.14 | 0.9 h |
| Opus 5.5 | single-round | 150 / 150 | ppa 233、lint 141、equiv 100、sim 23、value_range 3、value_at 1 | $21.57 | 1.7 h |
| Opus 5.5 | multi-round | 50 / 50 | ppa 592、equiv 460、lint 247、sim 133、value_range 3、value_at 1 | $63.07 | 4.9 h |
| Sonnet 5.5 | baseline | 0 / 150 | — | $4.51 | 0.5 h |
| Sonnet 5.5 | single-round | 7 / 150 | lint 8 | $5.00 | 0.5 h |
| Sonnet 5.5 | multi-round | 50 / 50 | ppa 268、equiv 193、lint 11、sim 2、synth 1 | $14.29 | 1.6 h |
| gpt-6-sol | baseline | 0 / 150 | — | in 4.7M / out 0.23M tokens | 1.9 h |
| gpt-6-sol | single-round | 150 / 150 | lint 442、ppa 389、sim 192、equiv 143、value_range 9 | in 34.2M / out 0.58M tokens | 5.5 h |
| gpt-6-sol | multi-round | 50 / 50 | lint 1239、ppa 1134、equiv 642、sim 233、value_at 2 | in 120.0M / out 1.29M tokens | 11.2 h |

Claude 的成本取自 CLI 回報的 `total_cost_usd`。multi-round 以 `--resume` 接續同一個 session，`total_cost_usd` 是整個 session 的累計值，因此每次 attempt 取最後一輪的值，不把各輪相加（相加會重複計算，得到 Opus $169.14、Sonnet $40.50）。Codex 不回報金額，只列 token（input 含 cached）；Codex `resume` 後每輪回報的 token 也是同一個 thread 的累計值，同樣取最後一輪（相加會得到 in 288.6M / out 3.5M）。Claude 的 `usage` 是每輪各自的值，所以照常相加。耗時是各 attempt 模型時間的總和，不含評分。

## 限制與偏離

- **沒有「多輪、無工具」對照組。** 目標文件只要求三組，所以 multi-round 的 PPA 改善同時包含「多輪探索的提示」與「工具」兩個因素。第 1 輪和 baseline 相當、前 3 輪就明顯改善，說明多輪是主因之一；工具佔多少比例無法從這組實驗分離。
- **multi-round 每題只跑 1 次**（使用者指示）。multi-round 的 pass@1 和 FUNCTION 只基於 50 次 attempt，不確定性比其他組（150 次）大。
- **用量上限造成的中斷與補跑。** 第一次完整執行有 701 / 900 次 attempt 撞到 Claude 帳號的 session 上限（HTTP 429），整批作廢，保留在 `_invalid_ratelimit_run1/` 但不列入計算。之後 multi-round 也多次撞到上限，改用 `rerun_multi.sh` 補跑：只要最後一輪的 CLI `result` 事件是 `is_error` 或帶 API 錯誤碼，就視為無效。每題取時間上最早的一個乾淨 attempt，選取規則只看時間，不看結果。被中斷的大多是較長的 attempt，可能讓乾淨樣本略偏向較快完成的 attempt。最終所有計入的 attempt 都沒有 API 錯誤、也沒有 timeout。
- **gpt-6-sol 的工具環境和 Claude 不完全相同。** Codex 透過新增的 `src/agent/codex.py` 執行：shell 關閉，寫檔用 `apply_patch`，IC 工具經由同一個 MCP server（排除 `show_wave`，tool timeout 1800 s）。Codex 一定會帶 code-mode `exec` 入口和 collaboration（sub-agent）工具，關不掉，但所有紀錄中都沒有任何 sub-agent 呼叫。Codex 沒有每次呼叫的預算上限，也沒有讀取已寫檔案的工具（multi-round 靠 session 歷史延續）。
- **Sonnet 在 single-round 幾乎不用工具**（7 / 150），所以它的 single-round 結果等於沒有工具的對照，不代表工具沒用。
- **不可綜合的題目。** 例如 `clkgenerator` 的規格要求 `initial` 和 `#` delay，功能正確但沒有 PPA，所以不會進入 frontier；這類題目計入 FUNCTION，但不計入 Pareto 候選數。
- **frontier 是抽樣得到的**，不代表全域最佳。PDAP 只在同一題內比較，數值越低不代表三個指標都改善。

## 檔案

| 路徑 | 內容 |
|---|---|
| `Report.md` | 主要比較、限制與逐題 Frontier Top 3 PDAP |
| `Reproduce.md`、`src/`、`analysis/`、`setup/`、`tests/` | 重現方法與實驗原始碼 |
| `problems/` | RTLLM 題目、參考解與測試 |
| 本機 `results/` | 1,050 次主實驗、450 次 shell 實驗、CLI 紀錄及 RTL 快照；未納入 PR |

## 附錄 A：逐題 Frontier Top 3 PDAP（目標文件格式）

每格點開可看逐題列表；N/A 表示該組在該題沒有有效候選。FUNCTION 為至少一個 snapshot 正確的題數。

### `claude-opus-5-5`

**實驗設定**：模型 `claude-opus-5-5`｜effort `medium`｜題數 `50`｜每題獨立重複 `3` 次（multi-round `1` 次）｜每輪預算 `$3.00`

| 實驗組 | Round | Success rate | FUNCTION | pass@1 | 每題 Pareto 候選數（平均） | Frontier Top 3：PDAP |
|---|---:|---:|---:|---:|---:|---|
| baseline (`notools`) | 1 | 100.0% | 41 / 50 | 78.7% | 1.94 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00301 / 0.00434<br>RAM：N/A<br>ROM：4.1e-07<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00339 / 0.00339 / 0.00366<br>adder_8bit：5.99e-05 / 5.99e-05 / 5.99e-05<br>adder_bcd：3.57e-05 / 3.57e-05<br>adder_pipe_64bit：0.625<br>alu：0.163 / 0.403 / 0.415<br>asyn_fifo：0.231 / 0.231 / 0.231<br>barrel_shifter：1.43e-05 / 1.43e-05 / 1.43e-05<br>calendar：0.00106 / 0.00106<br>clkgenerator：N/A<br>comparator_3bit：9.33e-07 / 9.33e-07<br>comparator_4bit：2.61e-06 / 2.61e-06<br>counter_12：7.73e-05 / 7.73e-05 / 7.73e-05<br>div_16bit：13 / 16.8<br>edge_detect：6.26e-06 / 6.26e-06 / 6.26e-06<br>fixed_point_adder：0.0234<br>fixed_point_substractor：0.0315<br>float_multi：N/A<br>freq_div：0.000214 / 0.000226 / 0.000226<br>freq_divbyeven：N/A<br>freq_divbyfrac：0.000763 / 0.000763 / 0.000763<br>freq_divbyodd：0.00219 / 0.00219 / 0.00219<br>fsm：1.11e-05<br>instr_reg：0.000344 / 0.000344<br>multi_16bit：0.351 / 0.358 / 0.358<br>multi_8bit：0.0823 / 0.0823 / 0.0823<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00138 / 0.00138 / 0.00138<br>multi_pipe_8bit：0.0498 / 0.0676<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：3.83e-05 / 3.83e-05 / 3.83e-05<br>sequence_detector：N/A<br>serial2parallel：0.00103<br>signal_generator：N/A<br>square_wave：0.00182 / 0.00182 / 0.00182<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：0.00316 / 0.00316 / 0.00316<br>up_down_counter：0.00448 / 0.00682 / 0.00682<br>width_8to16：0.00276 / 0.00313 / 0.00313</details> |
| single-round (`tools`) | 1 | 100.0% | 42 / 50 | 80.0% | 1.86 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00301 / 0.00333<br>RAM：N/A<br>ROM：4.1e-07 / 4.1e-07<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00325 / 0.00347<br>adder_8bit：6.2e-05 / 6.2e-05 / 6.2e-05<br>adder_bcd：3.42e-05 / 3.42e-05<br>adder_pipe_64bit：N/A<br>alu：0.115<br>asyn_fifo：0.231 / 0.231<br>barrel_shifter：1.43e-05 / 1.43e-05 / 1.43e-05<br>calendar：0.000989 / 0.00106<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：2.61e-06<br>counter_12：7.73e-05 / 7.73e-05 / 7.73e-05<br>div_16bit：14.1 / 16.6 / 16.6<br>edge_detect：6.26e-06 / 6.26e-06 / 6.26e-06<br>fixed_point_adder：0.0234<br>fixed_point_substractor：0.0157<br>float_multi：87.8<br>freq_div：0.000157<br>freq_divbyeven：N/A<br>freq_divbyfrac：0.000617<br>freq_divbyodd：0.00219 / 0.00219<br>fsm：9.88e-06 / 1.11e-05 / 1.37e-05<br>instr_reg：0.000344 / 0.000344<br>multi_16bit：0.169 / 0.252<br>multi_8bit：0.0202 / 0.0202 / 0.0202<br>multi_booth_8bit：0.0291<br>multi_pipe_4bit：0.00138 / 0.00138 / 0.00138<br>multi_pipe_8bit：0.0498 / 0.05<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：3.83e-05 / 3.83e-05 / 3.83e-05<br>sequence_detector：N/A<br>serial2parallel：0.00103<br>signal_generator：7.72e-05 / 7.72e-05 / 9.74e-05<br>square_wave：0.00182 / 0.00182 / 0.00182<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：0.00316 / 0.00316 / 0.00316<br>up_down_counter：0.00141<br>width_8to16：0.00313 / 0.00313 / 0.00313</details> |
| multi-round (`tools`) | 5 | 100.0% | 40 / 50 | 78.0% | 2.98 | <details><summary>50 題</summary>JC_counter：0.000203 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00181 / 0.00233 / 0.012<br>RAM：N/A<br>ROM：1.35e-07 / 1.35e-07 / 1.39e-07<br>accu：N/A<br>adder_16bit：0.000455 / 0.000455 / 0.000466<br>adder_32bit：0.0027 / 0.00284 / 0.00313<br>adder_8bit：5.15e-05 / 5.35e-05 / 8.7e-05<br>adder_bcd：3.42e-05 / 3.42e-05 / 3.42e-05<br>adder_pipe_64bit：N/A<br>alu：0.118 / 0.132 / 0.156<br>asyn_fifo：0.129 / 0.18 / 0.231<br>barrel_shifter：1.43e-05 / 1.43e-05 / 1.51e-05<br>calendar：0.000989 / 0.000996 / 0.00406<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：2.43e-06 / 2.58e-06 / 2.58e-06<br>counter_12：7.15e-05 / 7.73e-05 / 0.000264<br>div_16bit：4.6 / 4.63 / 16.8<br>edge_detect：1.98e-06 / 1.98e-06 / 2.08e-06<br>fixed_point_adder：0.0326 / 0.0329 / 0.0367<br>fixed_point_substractor：0.0171 / 0.0282 / 0.0282<br>float_multi：N/A<br>freq_div：1.6e-05 / 5.18e-05<br>freq_divbyeven：N/A<br>freq_divbyfrac：0.000481 / 0.000531 / 0.000538<br>freq_divbyodd：0.000693 / 0.000693<br>fsm：8.45e-06 / 1.43e-05<br>instr_reg：0.000344 / 0.000344 / 0.000344<br>multi_16bit：0.14 / 0.186 / 0.236<br>multi_8bit：0.0144 / 0.0149 / 0.0152<br>multi_booth_8bit：0.0151 / 0.0176 / 0.0185<br>multi_pipe_4bit：0.00115 / 0.00125 / 0.00137<br>multi_pipe_8bit：0.0578 / 0.0628 / 0.0735<br>parallel2serial：4.8e-05 / 4.97e-05 / 5.86e-05<br>pe：10.3 / 11.2 / 17.6<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：2.02e-05 / 2.44e-05 / 2.45e-05<br>sequence_detector：N/A<br>serial2parallel：0.000745 / 0.00082 / 0.00083<br>signal_generator：N/A<br>square_wave：0.000816 / 0.000884 / 0.00105<br>sub_64bit：0.0292 / 0.0293 / 0.0296<br>synchronizer：8.63e-06<br>traffic_light：0.000377 / 0.000377 / 0.000382<br>up_down_counter：0.00306 / 0.00314 / 0.00373<br>width_8to16：0.00258 / 0.00279 / 0.00313</details> |

### `claude-sonnet-5-5`

**實驗設定**：模型 `claude-sonnet-5-5`｜effort `medium`｜題數 `50`｜每題獨立重複 `3` 次（multi-round `1` 次）｜每輪預算 `$3.00`

| 實驗組 | Round | Success rate | FUNCTION | pass@1 | 每題 Pareto 候選數（平均） | Frontier Top 3：PDAP |
|---|---:|---:|---:|---:|---:|---|
| baseline (`notools`) | 1 | 99.3% | 33 / 50 | 64.0% | 1.58 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00348 / 0.00365 / 0.00458<br>RAM：N/A<br>ROM：4.1e-07 / 4.1e-07 / 4.1e-07<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00333 / 0.00333 / 0.00333<br>adder_8bit：6.19e-05 / 6.19e-05 / 6.19e-05<br>adder_bcd：3.42e-05<br>adder_pipe_64bit：N/A<br>alu：N/A<br>asyn_fifo：0.231<br>barrel_shifter：1.43e-05 / 1.43e-05 / 1.43e-05<br>calendar：0.00139 / 0.00139 / 0.00139<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：3.38e-06 / 3.38e-06 / 3.38e-06<br>counter_12：7.86e-05 / 7.86e-05 / 7.86e-05<br>div_16bit：10.6 / 36.4<br>edge_detect：6.26e-06 / 6.26e-06 / 6.26e-06<br>fixed_point_adder：0.029 / 0.029<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：0.000404 / 0.000404<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：3.35e-05<br>instr_reg：0.00031 / 0.00031 / 0.00031<br>multi_16bit：0.337 / 0.375<br>multi_8bit：0.0193 / 0.0193 / 0.0193<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00138 / 0.00138 / 0.00138<br>multi_pipe_8bit：N/A<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：N/A<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：9.74e-05 / 9.74e-05 / 9.74e-05<br>square_wave：0.00153<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：N/A<br>up_down_counter：0.00448 / 0.00448 / 0.00448<br>width_8to16：0.00289</details> |
| single-round (`tools`) | 1 | 100.0% | 36 / 50 | 64.7% | 1.78 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00435 / 0.00437 / 0.00437<br>RAM：N/A<br>ROM：4.1e-07 / 4.1e-07 / 4.1e-07<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00313 / 0.00313 / 0.00313<br>adder_8bit：6.19e-05 / 6.19e-05 / 6.19e-05<br>adder_bcd：3.42e-05 / 3.42e-05<br>adder_pipe_64bit：N/A<br>alu：0.471<br>asyn_fifo：0.232<br>barrel_shifter：1.43e-05 / 1.53e-05<br>calendar：0.00139 / 0.00139 / 0.00139<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：3.38e-06 / 3.38e-06 / 3.38e-06<br>counter_12：7.86e-05 / 7.86e-05 / 7.86e-05<br>div_16bit：12.3 / 12.3<br>edge_detect：6.26e-06 / 6.26e-06 / 6.26e-06<br>fixed_point_adder：0.0266<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：0.000404 / 0.000404 / 0.000404<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：3.35e-05 / 3.35e-05 / 3.35e-05<br>instr_reg：0.00031 / 0.00031 / 0.00031<br>multi_16bit：0.345 / 0.362<br>multi_8bit：0.0193 / 0.0193 / 0.0193<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00138 / 0.00138<br>multi_pipe_8bit：N/A<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：1.91e-06 / 2.04e-06 / 2.04e-06<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：3.83e-05 / 3.83e-05<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：9.74e-05 / 9.74e-05<br>square_wave：0.00182 / 0.00182 / 0.00182<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：N/A<br>up_down_counter：0.00448 / 0.00448 / 0.00448<br>width_8to16：0.00289 / 0.00289 / 0.00289</details> |
| multi-round (`tools`) | 5 | 99.6% | 29 / 50 | 56.0% | 2.14 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00248 / 0.00302 / 0.00302<br>RAM：N/A<br>ROM：4.1e-07 / 4.1e-07 / 4.1e-07<br>accu：N/A<br>adder_16bit：0.000409 / 0.000409 / 0.000409<br>adder_32bit：0.00313 / 0.00313 / 0.00313<br>adder_8bit：5.52e-05 / 5.52e-05 / 5.52e-05<br>adder_bcd：3.42e-05 / 3.42e-05 / 3.42e-05<br>adder_pipe_64bit：N/A<br>alu：N/A<br>asyn_fifo：N/A<br>barrel_shifter：N/A<br>calendar：0.00115 / 0.00117 / 0.00117<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：2.43e-06 / 2.61e-06 / 2.61e-06<br>counter_12：7.72e-05 / 7.72e-05 / 7.73e-05<br>div_16bit：14<br>edge_detect：2.28e-06 / 2.28e-06 / 2.28e-06<br>fixed_point_adder：N/A<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：9.91e-05 / 9.91e-05 / 0.000239<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：1.04e-05 / 1.04e-05 / 1.21e-05<br>instr_reg：0.000279 / 0.000279 / 0.000279<br>multi_16bit：0.253 / 0.253 / 0.253<br>multi_8bit：0.0149 / 0.0149 / 0.0166<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：N/A<br>multi_pipe_8bit：N/A<br>parallel2serial：4.18e-05 / 5.86e-05 / 5.86e-05<br>pe：8.96 / 14.8 / 18<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：N/A<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：7.72e-05 / 7.72e-05 / 7.72e-05<br>square_wave：0.00109 / 0.00109<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：N/A<br>up_down_counter：0.00148 / 0.00148 / 0.00148<br>width_8to16：0.00276 / 0.00276 / 0.00288</details> |

### `gpt-6-sol`

**實驗設定**：模型 `gpt-6-sol`｜effort `medium`｜題數 `50`｜每題獨立重複 `3` 次（multi-round `1` 次）｜每輪預算 無（Codex CLI 無預算參數）

| 實驗組 | Round | Success rate | FUNCTION | pass@1 | 每題 Pareto 候選數（平均） | Frontier Top 3：PDAP |
|---|---:|---:|---:|---:|---:|---|
| baseline (`notools`) | 1 | 100.0% | 31 / 50 | 58.7% | 1.34 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00453<br>RAM：N/A<br>ROM：N/A<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00313 / 0.0032<br>adder_8bit：5.99e-05<br>adder_bcd：3.57e-05 / 3.57e-05<br>adder_pipe_64bit：N/A<br>alu：N/A<br>asyn_fifo：N/A<br>barrel_shifter：N/A<br>calendar：0.00139 / 0.00139 / 0.00158<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：2.61e-06 / 2.61e-06 / 2.61e-06<br>counter_12：7.86e-05 / 7.86e-05 / 7.86e-05<br>div_16bit：N/A<br>edge_detect：2.28e-06<br>fixed_point_adder：0.0404 / 0.0404<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：0.000404 / 0.000404 / 0.000404<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：1.21e-05 / 1.21e-05<br>instr_reg：0.00031 / 0.00031 / 0.00031<br>multi_16bit：0.172<br>multi_8bit：0.0202 / 0.0202 / 0.0202<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00138 / 0.00138<br>multi_pipe_8bit：N/A<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：3.83e-05 / 3.83e-05 / 3.83e-05<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：N/A<br>square_wave：0.00182 / 0.00182<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：0.00316 / 0.00316<br>up_down_counter：0.00161<br>width_8to16：0.00289 / 0.00289 / 0.00289</details> |
| single-round (`tools`) | 1 | 99.3% | 30 / 50 | 57.3% | 1.22 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00212 / 0.00363<br>RAM：N/A<br>ROM：N/A<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00313 / 0.00313 / 0.00325<br>adder_8bit：6.1e-05 / 6.1e-05<br>adder_bcd：3.57e-05 / 3.57e-05 / 3.57e-05<br>adder_pipe_64bit：N/A<br>alu：N/A<br>asyn_fifo：N/A<br>barrel_shifter：N/A<br>calendar：0.00113 / 0.00114<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：2.58e-06 / 2.58e-06<br>counter_12：7.86e-05 / 7.86e-05 / 7.86e-05<br>div_16bit：N/A<br>edge_detect：2.28e-06<br>fixed_point_adder：N/A<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：0.000241<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：1.06e-05 / 1.21e-05 / 1.21e-05<br>instr_reg：0.000279<br>multi_16bit：0.202 / 0.25<br>multi_8bit：0.0149 / 0.0149 / 0.0149<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00115<br>multi_pipe_8bit：N/A<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：2.53e-05 / 2.53e-05 / 3.83e-05<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：N/A<br>square_wave：0.00125<br>sub_64bit：0.0184 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：0.00316<br>up_down_counter：0.00148 / 0.00148 / 0.00148<br>width_8to16：0.0021</details> |
| multi-round (`tools`) | 5 | 99.2% | 28 / 50 | 54.0% | 1.76 | <details><summary>50 題</summary>JC_counter：0.000445 / 0.000454 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00172 / 0.00215 / 0.00297<br>RAM：N/A<br>ROM：5.59e-10<br>accu：N/A<br>adder_16bit：0.000427 / 0.000434 / 0.00044<br>adder_32bit：0.00291 / 0.00324 / 0.00325<br>adder_8bit：5.52e-05 / 5.67e-05 / 5.69e-05<br>adder_bcd：3.34e-05 / 3.45e-05 / 3.61e-05<br>adder_pipe_64bit：N/A<br>alu：N/A<br>asyn_fifo：N/A<br>barrel_shifter：N/A<br>calendar：0.000836 / 0.00105<br>clkgenerator：N/A<br>comparator_3bit：8.35e-07 / 8.35e-07 / 1.05e-06<br>comparator_4bit：2.43e-06 / 2.56e-06 / 2.56e-06<br>counter_12：6.53e-05 / 7.58e-05 / 7.73e-05<br>div_16bit：N/A<br>edge_detect：1.98e-06 / 1.98e-06 / 1.98e-06<br>fixed_point_adder：N/A<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：0.000154<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：6.44e-06 / 9.15e-06 / 1.32e-05<br>instr_reg：0.000277 / 0.000278 / 0.000279<br>multi_16bit：0.245 / 0.256 / 0.305<br>multi_8bit：0.0111 / 0.0112 / 0.012<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.000933 / 0.00115 / 0.00144<br>multi_pipe_8bit：N/A<br>parallel2serial：4.13e-05 / 5.37e-05 / 6.55e-05<br>pe：10.2 / 11 / 14.7<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.56e-05 / 1.56e-05 / 1.56e-05<br>ring_counter：1.29e-05 / 2.02e-05 / 2.58e-05<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：N/A<br>square_wave：N/A<br>sub_64bit：0.0294 / 0.0296 / 0.0296<br>synchronizer：9.64e-06<br>traffic_light：N/A<br>up_down_counter：0.00148 / 0.00149 / 0.0017<br>width_8to16：0.0021 / 0.00279</details> |

## 附錄 B：baseline + shell

2026-10-04 另外跑的第 4 組：不給 IC 工具，但兩個後端都可以用 shell。其他設定與 baseline 相同（每題 3 次、1 輪）。

### 設定

同 baseline，但兩個後端都開 shell：Claude `--tools Write,Bash`；Codex 不關 `shell_tool`。shell 的 PATH 固定為 `/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin`，有 iverilog 13.0、verilator、yosys、python3，沒有 `ic` 和 openroad。每題 3 次；Claude Code CLI 2.1.289

### 主要指標

PDAP / baseline 以無 shell 的 baseline 為分母，且**沿用不設限的舊定義**（每題取該組最佳 PDAP 相除後取幾何平均），與「比較表格」改用的三項全贏口徑不同，兩者不可直接相比。候選數欄同樣是「每題 Pareto 候選數（平均）」的舊定義。

| 模型 | 實驗組 | Round | Success rate | pass@3 | pass@1 | 每題 Pareto 候選數（平均） | PDAP / baseline |
|---|---|---|---|---|---|---|---|
| Opus 5.5 | baseline + shell (`notools` + Bash) | 1 | 100.0% | 88.0% | 84.0% | 2.00 | 1.011× (n=38) |
| Sonnet 5.5 | baseline + shell (`notools` + Bash) | 1 | 100.0% | 66.0% | 62.7% | 1.70 | 0.996× (n=30) |
| gpt-6-sol | baseline + shell (`notools` + Bash) | 1 | 98.7% | 60.0% | 55.3% | 1.32 | 1.062× (n=27) |

### 正確性

| 模型 | 實驗組 | Attempts | pass@1 | pass@3 | 任一輪正確 | FUNCTION |
|---|---|---|---|---|---|---|
| Opus 5.5 | baseline + shell | 150 | 84.0% | 88.0% | 84.0% | 44 / 50 |
| Sonnet 5.5 | baseline + shell | 150 | 62.7% | 66.0% | 62.7% | 33 / 50 |
| gpt-6-sol | baseline + shell | 150 | 55.3% | 60.0% | 55.3% | 30 / 50 |

### PPA：同題最佳 PDAP 相對 baseline

| 模型 | 比較 | PDAP / baseline（幾何平均） | n | 勝 / 平 / 負 | 平均候選數 |
|---|---|---|---|---|---|
| Opus 5.5 | baseline + shell | 1.011× | 38 | 7 / 22 / 9 | 2.00 |
| Sonnet 5.5 | baseline + shell | 0.996× | 30 | 2 / 23 / 5 | 1.70 |
| gpt-6-sol | baseline + shell | 1.062× | 27 | 1 / 22 / 4 | 1.32 |

### PPA：直接對比 IC 工具組

比值為「shell baseline 最佳 PDAP ÷ 該組最佳 PDAP」，大於 1 代表 shell baseline 較差；勝 / 平 / 負以 shell baseline 的角度計算。

| 模型 | 對比 | PDAP 比值（幾何平均） | n | shell baseline 勝 / 平 / 負 |
|---|---|---:|---:|---|
| Opus 5.5 | single-round（IC 工具，1 輪） | 1.056× | 40 | 5 / 19 / 16 |
| Opus 5.5 | multi-round（IC 工具，5 輪） | 1.721× | 38 | 0 / 3 / 35 |
| Sonnet 5.5 | single-round | 0.982× | 31 | 3 / 27 / 1 |
| Sonnet 5.5 | multi-round | 1.334× | 27 | 1 / 7 / 19 |
| gpt-6-sol | single-round | 1.213× | 26 | 1 / 12 / 13 |
| gpt-6-sol | multi-round | 1.506× | 25 | 0 / 1 / 24 |

Sonnet 的 single-round 幾乎沒用 IC 工具，所以兩者接近。

### 工具使用與成本

| 模型 | 實驗組 | 呼叫 IC 工具的 attempt | IC 工具呼叫次數 | 成本 / token | 模型耗時 |
|---|---|---|---|---|---|
| Opus 5.5 | baseline + shell | 0 / 150（用了 Bash：107 / 150） | Bash 273 | $17.69 | 1.6 h |
| Sonnet 5.5 | baseline + shell | 0 / 150（用了 Bash：16 / 150） | Bash 17 | $4.98 | 0.5 h |
| gpt-6-sol | baseline + shell | 0 / 150（用了 Bash：150 / 150） | Bash 856 | in 10.8M / out 0.35M tokens | 3.0 h |

### 逐題 Frontier Top 3 PDAP

格式同附錄 A。

### `claude-opus-5-5`

| 實驗組 | Round | Success rate | FUNCTION | pass@1 | 每題 Pareto 候選數（平均） | Frontier Top 3：PDAP |
|---|---|---|---|---|---|---|
| baseline + shell (`notools` + Bash) | 1 | 100.0% | 44 / 50 | 84.0% | 2.00 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00387<br>RAM：N/A<br>ROM：4.1e-07<br>accu：0.00566<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00351 / 0.00351 / 0.00357<br>adder_8bit：6.2e-05 / 6.2e-05 / 6.2e-05<br>adder_bcd：3.57e-05 / 3.57e-05<br>adder_pipe_64bit：N/A<br>alu：0.127 / 0.132<br>asyn_fifo：0.231 / 0.231 / 0.231<br>barrel_shifter：1.43e-05 / 1.43e-05 / 1.43e-05<br>calendar：0.00106 / 0.00106<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07<br>comparator_4bit：2.61e-06 / 2.61e-06 / 2.61e-06<br>counter_12：7.73e-05 / 7.73e-05 / 7.73e-05<br>div_16bit：16.2 / 16.8<br>edge_detect：6.26e-06 / 6.26e-06 / 6.26e-06<br>fixed_point_adder：0.0405 / 0.0411 / 0.0468<br>fixed_point_substractor：0.0321 / 0.0324<br>float_multi：3.8 / 4.73<br>freq_div：0.000226 / 0.000237<br>freq_divbyeven：N/A<br>freq_divbyfrac：0.000763 / 0.000763 / 0.000763<br>freq_divbyodd：0.00219 / 0.00219 / 0.00219<br>fsm：1.11e-05<br>instr_reg：0.000344<br>multi_16bit：0.289 / 0.289 / 0.289<br>multi_8bit：0.0823 / 0.0823 / 0.0823<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00138 / 0.00138 / 0.00138<br>multi_pipe_8bit：0.0735<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：3e-06<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：3.83e-05 / 3.83e-05 / 3.83e-05<br>sequence_detector：N/A<br>serial2parallel：0.000973<br>signal_generator：7.72e-05 / 7.72e-05 / 9.74e-05<br>square_wave：0.00154 / 0.00182 / 0.00182<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：0.00208<br>up_down_counter：0.00432 / 0.00448 / 0.00448<br>width_8to16：0.00276 / 0.00313 / 0.00313</details> |

### `claude-sonnet-5-5`

| 實驗組 | Round | Success rate | FUNCTION | pass@1 | 每題 Pareto 候選數（平均） | Frontier Top 3：PDAP |
|---|---|---|---|---|---|---|
| baseline + shell (`notools` + Bash) | 1 | 100.0% | 33 / 50 | 62.7% | 1.70 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00369 / 0.00425 / 0.00492<br>RAM：N/A<br>ROM：4.1e-07 / 4.1e-07 / 4.1e-07<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00313 / 0.0044<br>adder_8bit：6.19e-05 / 6.37e-05 / 6.37e-05<br>adder_bcd：3.42e-05 / 3.42e-05 / 3.42e-05<br>adder_pipe_64bit：N/A<br>alu：N/A<br>asyn_fifo：0.232<br>barrel_shifter：1.43e-05 / 1.43e-05 / 1.43e-05<br>calendar：0.00139 / 0.00139 / 0.00139<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：3.38e-06 / 3.38e-06 / 3.38e-06<br>counter_12：7.86e-05 / 7.86e-05 / 7.86e-05<br>div_16bit：12.3 / 12.3 / 12.3<br>edge_detect：6.26e-06 / 6.26e-06 / 6.26e-06<br>fixed_point_adder：N/A<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：0.000404 / 0.000404 / 0.000404<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：2.5e-05<br>instr_reg：0.00031 / 0.00031 / 0.00031<br>multi_16bit：0.345 / 0.345 / 0.375<br>multi_8bit：0.0193 / 0.0193 / 0.0193<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00138 / 0.00138 / 0.00138<br>multi_pipe_8bit：N/A<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：2.04e-06<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：N/A<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：9.74e-05 / 9.74e-05<br>square_wave：0.00154 / 0.00154 / 0.00182<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：N/A<br>up_down_counter：0.00448 / 0.00448 / 0.00448<br>width_8to16：0.00289 / 0.00289 / 0.00289</details> |

### `gpt-6-sol`

| 實驗組 | Round | Success rate | FUNCTION | pass@1 | 每題 Pareto 候選數（平均） | Frontier Top 3：PDAP |
|---|---|---|---|---|---|---|
| baseline + shell (`notools` + Bash) | 1 | 98.7% | 30 / 50 | 55.3% | 1.32 | <details><summary>50 題</summary>JC_counter：0.00154 / 0.00154 / 0.00154<br>LFSR：N/A<br>LIFObuffer：0.00549<br>RAM：N/A<br>ROM：N/A<br>accu：N/A<br>adder_16bit：0.000466 / 0.000466 / 0.000466<br>adder_32bit：0.00313 / 0.00323<br>adder_8bit：5.99e-05 / 5.99e-05 / 5.99e-05<br>adder_bcd：3.57e-05<br>adder_pipe_64bit：N/A<br>alu：N/A<br>asyn_fifo：N/A<br>barrel_shifter：N/A<br>calendar：0.00158 / 0.00158 / 0.00158<br>clkgenerator：N/A<br>comparator_3bit：9.28e-07 / 9.28e-07 / 9.28e-07<br>comparator_4bit：2.61e-06 / 2.61e-06 / 2.61e-06<br>counter_12：7.86e-05 / 7.86e-05 / 7.86e-05<br>div_16bit：N/A<br>edge_detect：6.26e-06 / 6.26e-06 / 6.26e-06<br>fixed_point_adder：0.0404<br>fixed_point_substractor：N/A<br>float_multi：N/A<br>freq_div：0.000404 / 0.000404 / 0.000404<br>freq_divbyeven：N/A<br>freq_divbyfrac：N/A<br>freq_divbyodd：N/A<br>fsm：1.21e-05<br>instr_reg：0.00031 / 0.00031 / 0.00031<br>multi_16bit：0.251 / 0.275<br>multi_8bit：0.0202 / 0.0202 / 0.0202<br>multi_booth_8bit：N/A<br>multi_pipe_4bit：0.00138 / 0.00138<br>multi_pipe_8bit：N/A<br>parallel2serial：5.86e-05 / 5.86e-05 / 5.86e-05<br>pe：18 / 18 / 18<br>pulse_detect：N/A<br>radix2_div：N/A<br>right_shifter：1.88e-05 / 1.88e-05 / 1.88e-05<br>ring_counter：3.83e-05 / 3.83e-05 / 3.83e-05<br>sequence_detector：N/A<br>serial2parallel：N/A<br>signal_generator：N/A<br>square_wave：N/A<br>sub_64bit：0.0311 / 0.0311 / 0.0311<br>synchronizer：1.04e-05 / 1.04e-05 / 1.04e-05<br>traffic_light：0.00316<br>up_down_counter：0.00148<br>width_8to16：0.00289 / 0.00289 / 0.00289</details> |

### 限制

- **baseline + shell 是另一次執行。** 它在 2026-10-04 跑，其他組在 2026-10-03 跑；模型 ID、題目、提示詞、評分都相同，但 Claude Code CLI 從 2.1.288 自動更新成 2.1.289。
- **baseline + shell 的限制。**
  - 題目資料沒有隔離：shell 讀得到題目目錄與參考解。事後檢查所有 shell 指令，沒有任何一條碰到題目目錄、`~/.ab_private` 或 `verified_*`；紀錄中出現 RTLLM 字樣的地方，都是模型說「憑記憶參考」。
  - Claude 的 Bash 可以連網，Codex 的 shell 在 `workspace-write` sandbox 下不能連網。
  - Claude 的 shell baseline 只有 `Write,Bash`，沒有 Read / Edit；不過 Bash 可以讀檔、改檔。

### 檔案

| 路徑 | 內容 |
|---|---|
| `Report.md` | baseline + shell 的比較結果 |
| 本機 `results/shell/` | 450 次乾淨 attempt、Top 3 RTL 快照及重新評分資料；未納入 PR |
| 本機封存的 `results/shell/` 其他資料 | 未經要求就跑的 shell 版 tools 組與 multi-round（Sonnet tools 組 150 次、Opus tools 組與 Sonnet multi-round 各一部分），未納入本報告或 PR |
