# NishimoriLine — 2D Ising Monte Carlo tools

本仓库提供一个基于 TUI 的 2D Ising Model 扫描主程序，以及若干离线分析脚本：

- 主程序：在 \(T\) 方向做 MC 扫描，自动提取临界区间并做 log-log 拟合
- 重分析脚本：对已有的单次 log-log 结果做离线分析
- 聚合脚本：在 \(p\)–\(T_c\) 平面上对多次结果做统计与可视化

下面是各个程序的使用方式与数据目录结构。

## 1. 主程序：TUI 扫描与自动分析

在仓库根目录运行：

```bash
cargo run
```

功能概览：

- Setup 界面：
  - 编辑模型参数、MC 参数、当前扫描区间
  - `c`：从已有 `data/` 目录下的 `summary.txt` 复制参数（按时间戳选择）
  - `Enter`：开始 Step 1 的宽区间扫描
- Step 1：
  - 扫描 \(T_\text{start} \to T_\text{end}\)，估计 \(C(T)\) 与 \(\chi(T)\) 的主峰与次峰
  - 自动给出包络区间（用于 \(T\) 扫描）和重叠区间（用于 \(T_c\) 拟合）
- Step 2：
  - 在选定的 \(T_c\) 区间上做 log-log 拟合，给出 `Tc_best` 与临界指数
  - 所有结果写入 `data/loglog_singleProfile_YYYYMMDD_HHMMSS/` 中

主程序会自动在 `data/` 下生成两类子目录：

- `ising_results_YYYYMMDD_HHMMSS/`：Step 1 宽区间扫描的物理量结果
- `loglog_singleProfile_YYYYMMDD_HHMMSS/`：对应 Step 2 log-log 拟合结果

每个 `loglog_singleProfile_*` 目录下会有：

- `summary.txt`：记录本次模拟的全部参数与最佳 \(T_c\)
- `loglog_singleProfile_scan.csv`、`loglog_singleProfile_tc_scan.csv`
- `loglog_singleProfile_loglog_detailed.html`：包含 log-log 拟合细节

## 2. 重分析脚本：`reanalysis.py`

`reanalysis.py` 用于对已有的某次 log-log 结果做离线 Step 2 分析（无需重新跑 MC）。

基本用法（在仓库根目录）：

```bash
python reanalysis.py data/loglog_singleProfile_YYYYMMDD_HHMMSS
```

其中 `data/loglog_singleProfile_YYYYMMDD_HHMMSS` 是你想重分析的目标目录。
脚本会读取该目录中的 CSV/HTML/summary，重新扫描 \(T_c\) 并输出更新后的结果文件。

## 3. 聚合脚本：`auto_aggregation.rs`

`auto_aggregation` 用于在 \(p\)–\(T_c\) 平面上对最近 N 次 log-log 结果做统计，并输出图像与文本 Summary。

### 3.1 运行方式

交互方式（推荐）：

```bash
cargo run --bin auto-aggregation
```

程序会提示输入要聚合的最近 N 条 log-log 结果：

```text
Enter number of recent runs N to aggregate:
```

例如输入 `20`，则会从 `candidate_data/` 下按时间戳逆序选出最近的 20 个
`loglog_singleProfile_YYYYMMDD_HHMMSS` 目录，读取其中的：

- `p`：无序度
- `Tc_best`：最佳临界温度

并在终端打印：

- 每个样本点的 `(dir, p, Tc_best)`
- 按 p 分组后的统计量：样本数、平均 \(T_c\)、方差

随后按回车，进入输出阶段。

非交互方式（可选，用于脚本化调用）：

```bash
cargo run --bin auto-aggregation -- 20
```

这里 `20` 就是 N，程序不会等待你再次确认，直接输出结果。

### 3.2 data2 目录结构（按脚本 + 时间戳分类）

聚合结果会统一写入 `data2/` 下的子目录，命名格式为：

```text
data2/auto_aggregation_YYYYMMDD_HHMMSS/
```

每个这样的子目录中包含：

- `tp_aggregation.png`
  - 横轴：p
  - 纵轴：Tc_best
  - 蓝色点：每一次 loglog 结果给出的 `(p, Tc_best)` 样本点
  - 红色点：同一 p 下所有样本的 Tc 平均值
- `tp_aggregation_summary.txt`
  - 每个样本点的 `(dir, p, Tc_best)`
  - 按 p 归类后的：样本数量、平均 \(T_c\)、方差

后续如果新增其他离线脚本，也可以统一将输出放在 `data2/<脚本名>_YYYYMMDD_HHMMSS/` 的结构下，方便浏览与管理。
