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

### 3.3 `candidate_data/` 的使用方式

为避免在大杂烩的 `data/` 目录中扫描所有结果，聚合脚本不会直接读取 `data/`，而是只读取：

- `candidate_data/loglog_singleProfile_YYYYMMDD_HHMMSS/`

推荐的用法是：

- 正常跑主程序或批量脚本，生成 `data/` 或 `data_batch/` 下的 `loglog_singleProfile_*` 目录
- 从中挑选你认为“值得聚合”的那一批，复制（或移动）到 `candidate_data/` 下
- 然后再运行 `auto-aggregation`，只会对这些候选样本做 \(p\)–\(T_c\) 聚合

这样可以避免过期/试跑数据干扰统计结果。

## 4. 批量 Tc–p 扫描：`batch-input`

`batch-input` 用于在一批不同的 \(p\) 上重复跑“单次 \(T\) 扫描 + log-log 分析”，并记录每个 \(p\) 的 \(T_c\) 结果。

### 4.1 运行方式

在仓库根目录运行：

```bash
cargo run --bin batch-input
```

会进入一个 TUI 界面，包含三组参数：

- Model Parameters：如 L、J、外场 H
- Scan Parameters：\(T_\text{start}\)、\(T_\text{end}\)、\(T_\text{step}\)
- MC + Batch Parameters：
  - MC steps、Therm steps、Stride、Disorder samples
  - `p start`、`p end`、`p step`

键盘操作：

- `↑` / `↓`：在字段之间移动光标
- 直接输入数字：编辑当前字段
- `Backspace`：删除一位
- `o`：切换异常点过滤开关（Outlier filter，off / open）
- `w`：切换 log-log 窗口模式（见下）
- `Enter`：开始批量运行
- `q`：退出

### 4.2 log-log 窗口模式：A / B

在批量模式中，每个 \(p\) 的 log-log 拟合窗口可以有两种策略：

- 模式 A：`window mode = A: primary`
  - 对每个 \(p\)，使用自动分析得到的 primary 窗口
  - 不显示、也不编辑 \(T\) / \(T_c\) 窗口数值
- 模式 B：`window mode = B: fixed`
  - 所有 \(p\) 公用同一组窗口：
    - `T win min`，`T win max`
    - `Tc win min`，`Tc win max`
  - 这四个字段可在 TUI 中编辑，默认值为：
    - \(T \in [2.0, 2.45]\)
    - \(T_c \in [2.25, 2.45]\)

在界面顶部会实时显示当前模式与开关状态，例如：

```text
mode=A: primary  outlier=off  keys: ↑↓ move  Enter start  q quit  o outlier  w window mode
```

### 4.3 输出目录结构：`data_batch/`

批量运行的所有输出统一写入：

```text
data_batch/batch_YYYYMMDD_HHMMSS/
```

结构示例：

```text
data_batch/
  batch_20260222_123456/
    loglog_singleProfile_20260222_123500/   # 对应某个 p 的单次结果
    loglog_singleProfile_20260222_123700/
    ...
```

每个 `loglog_singleProfile_*` 子目录的内容与主程序的输出格式一致，包含 `summary.txt`、扫描 CSV、以及 HTML 详细结果。

如果你希望将某些批量结果纳入 \(p\)–\(T_c\) 聚合，只需从这些 `loglog_singleProfile_*` 目录中挑选若干，复制到 `candidate_data/` 下，再运行 `auto-aggregation` 即可。


你问得非常到位。
这里确实容易逻辑混淆。

我们把三件事彻底分开：

1. **Nishimori line 的定义是什么**
2. **内能公式从哪里来**
3. **数值上到底怎么“跑”出来**

---

# 一、Nishimori line 的真正定义

Nishimori line 不是通过内能定义的。

它的定义是：

[
\beta J = K_p
]

也就是让

[
P(J_{ij}) \propto e^{K_p J_{ij}}
]

和

[
e^{\beta J_{ij} S_i S_j}
]

结构匹配。

在你约定（(p) 为反铁磁比例）下：

[
e^{-2\beta J} = \frac{p}{1-p}
]

这是**定义式**。

它来自：

> 要求 disorder 分布和 Boltzmann 权重在 gauge 变换下协变。

内能公式是**推论**，不是定义。

你说得完全对。

---

# 二、内能公式从哪里来？

在 Nishimori line 上可以严格证明：

[
U = -N_b J \tanh(\beta J)
]

其中：

* (N_b) 是总 bond 数
* 这是 disorder-averaged internal energy

如果你写成 **每个 bond 的平均能量**：

[
u_b = - J \tanh(\beta J)
]

如果写成 **每个 spin 的能量**：

[
u = \frac{U}{N}
]

那要乘上配位数因子：

[
u = - \frac{z}{2} J \tanh(\beta J)
]

其中 (z) 是配位数。

所以：

> 公式本质是每 bond 的平均能量。

---

# 三、那 Monte Carlo 怎么“找” Nishimori line？

分两种情况。

---

## 情况 A：你已经知道 p

那根本不需要 MC 找。

直接用：

[
T(p) = \frac{2J}{\ln \frac{1-p}{p}}
]

这就是 Nishimori line。

MC 没有必要。

---

## 情况 B：你假装不知道公式

这时可以这样做：

### 步骤

1. 固定一个 p
2. 对多个温度做 MC
3. 计算 disorder average 的内能

也就是：

* 每组 (J_{ij}) 做一次 MC
* 算热平均
* 再对 disorder 取平均

得到：

[
u_{MC}(T,p)
]

然后寻找满足：

[
u_{MC}(T,p) = - \frac{z}{2} J \tanh(\beta J)
]

