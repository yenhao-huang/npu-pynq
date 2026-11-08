goal
* 測試使用工具是否有提升 PPA

Models
* opus 5.5, medium effort
* gpt-6-sol, medium effort
* sonnet 5.5, medium effort

repetition: 每題獨立重複 3 次 (`--repetitions 3`)，除了 multi-round 除外
round: 每次對話輪數 (`--rounds`)，baseline 與 single-round 為 1，multi-round 暫設 5

實驗組合
* baseline: 一輪對話，得到解答
* single-round: 一輪對話，能使用工具，得到解答
* multi-round: 多輪對話，能使用工具，得到解答

目標
* syntax 合法率
* Functional correctness: pass@1 and pass@3
* 找出 pareto frontier (指標: area_um2、power_w、delay_ns)

工具包含
| 工具 | 用途 |
|---|---|
| `lint` | RTL 語法與靜態檢查 |
| `sim` | 執行模擬 |
| `signals` | 列出波形訊號 |
| `first_mismatch` | 尋找首次訊號不一致 |
| `value_at` | 查詢指定時間的訊號值 |
| `value_range` | 查詢時間區間內的訊號值 |
| `synth` | 綜合設計 |
| `ppa` | 量測功耗、效能與面積 |
| `equiv` | 等價性檢查 |

modules 位置: src/

Notes:
* 實驗要建立在 modules 上，能做的只有設定參數 (e.g., 換模型、換組合)
* 實驗結果寫在 evidence/
* 結束給我 Report.md 與 Reproduce.md
    * Report.md 要有比較表格
    * pareto frontier 前 3 名的 reproduce
* 先不討論 directed
* 三組一起比較
* PDAP = power_w × delay_ns × area_um2. Use it to rank Pareto candidates and summarize relative performance against baseline; it does not determine frontier membership.

比較表格
**實驗設定**：模型 `___`｜effort `medium`｜題數 `___`｜每題獨立重複 `3` 次｜每輪預算 `___`

| 實驗組 | Round | Success rate | pass@3 | pass@1 | 每題 Pareto 候選數（平均） | PDAP / baseline |
|---|---:|---:|---:|---:|---:|---|
| baseline (`notools`) | 1 | —% | —% | —% | — | —× (n=—) |
| single-round (`tools`) | 1 | —% | —% | —% | — | —× (n=—) |
| multi-round (`tools`) | 3 | —% | —% | —% | — | —× (n=—) |

* Pareto 候選數只作參考，數量較多不能直接證明使用工具較好。
* PDAP / baseline: for each problem, select each group's lowest PDAP among functionally correct candidates with valid PPA across all attempts and rounds. Divide the group's best PDAP by baseline's best PDAP for the same problem, then take the geometric mean of these ratios. Lower is better; 1.00× equals baseline.
* Show the aggregate ratio and paired problem count as `0.879× (n=38)`. Include only problems with valid candidates in both groups; show N/A if none qualify. Comparison sets may differ between groups. Do not pool or directly average raw PDAP across problems.
* Keep per-problem raw PDAP, frontier Top 3, and reproduction paths in `evidence/summary.json`; omit per-problem Top 3 listings from the comparison table. Retain Top 3 reproduction details in Reproduce.md.
* 同一題的 PDAP 越低，代表乘積表現越好，不代表 Power、Delay、Area 三項都改善；搭配 pass@3 與 pass@1 一起比較。

* pass@3: the percentage of problems with at least one functionally correct final answer across three independent attempts. For multi-round attempts, evaluate only the final round; an earlier passing round does not count if the final answer fails. Report N/A until all three attempts per problem are available.
* pass@1: compute the fraction of functionally correct final answers across independent attempts for each problem, then average across problems.
