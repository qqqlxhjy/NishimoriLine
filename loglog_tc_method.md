## 2D Ising 模型临界点 Tc 的 log-log 标度分析方法（方法二）

### 输入数据
- 来自一次温度扫描的观测数据
  - 温度 \(T_i\)
  - 每个温度下的平均磁化强度 \(M_i \approx \langle |M| \rangle / N\)
- 数据通常以 CSV 形式存储，列为
  - `temperature`
  - `m_abs_per_spin`
  - 以及其他量（能量、比热等，Tc 分析只用到前两列）

### 理论基础
- 2D Ising 模型在 \(T_c\) 附近的标度关系
  \[
  M(T) \sim A (T_c - T)^{\beta}, \quad T \to T_c^{-}
  \]
- 取对数得到线性关系
  \[
  \ln M = \ln A + \beta \ln (T_c - T)
  \]
- 若给定正确的 \(T_c\)，则在 \(T < T_c\) 的临界区域，图像
  \[
  x = \ln(T_c - T), \quad y = \ln M
  \]
  应该近似落在一条直线上，斜率即为临界指数 \(\beta\)。

### 方法二的核心思路（扫描 Tc 的 log-log 法）
1. 先用蒙特卡洛模拟做一次温度扫描，生成 \(T_i, M_i\) 数据。
2. 设定一个候选 Tc 的搜索区间
   - \(T_c \in [T_{c,\min}, T_{c,\max}]\)
   - 给定步长 \(\Delta T_c\)。
3. 对每一个候选 \(T_c\)：
   - 只考虑满足条件的点
     - \(T_i < T_c\)
     - \(T_i\) 处于给定分析区间 \([T_{\min}, T_{\max}]\) 内
     - \(M_i > 0\)
   - 对这些点构造
     - \(x_i = \ln(T_c - T_i)\)
     - \(y_i = \ln M_i\)
   - 对 \((x_i, y_i)\) 做线性回归
     - 得到斜率 `slope` 和截距 `intercept`
     - 计算拟合优度 \(R^2\)
   - 物理约束
     - 斜率必须为正：`slope > 0`，因为 \(\beta > 0\)
     - \(R^2\) 需在合理范围内：\(0 < R^2 \le 1\)
4. 在所有候选 \(T_c\) 中，选出满足物理约束且 \(R^2\) 最大的那一个
   - 该候选的
     - `tc` 即为估计的临界温度 \(T_c\)
     - `beta = slope` 即为估计的临界指数 \(\beta\)

### 代码结构要点（homework12 / Isingmodel）
- 主程序 `main.rs`
  - 子命令 `Scan` 负责执行温度扫描并输出 CSV
  - 子命令 `Tc` 负责 log-log 法 Tc 分析
    - 解析 Tc 分析参数：`t_min, t_max, tc_min, tc_max, tc_step, data_file`
    - 调用 `LogLogAnalyzer` 进行扫描和拟合
    - 输出最佳 Tc 与 β，并生成结果文件与图像

- 模块 `beta_trial.rs`
  - 结构 `LogLogAnalysisParameters`
    - 记录分析温度窗口 `[t_min, t_max]`、候选 Tc 范围 `[tc_min, tc_max]`、步长 `tc_step` 以及数据路径
  - 结构 `LogLogResult`
    - 存储一次候选 Tc 的拟合结果：`tc, beta(=slope), r_squared, intercept, fit_points, is_valid`
  - 结构 `LogLogAnalyzer`
    - 字段：`temperatures`、`magnetizations`、`params`
    - `new`：从 CSV 中读取 \(T_i, M_i\)
    - `analyze`：在给定 Tc 范围内扫描，调用 `evaluate_tc_trial(tc)`
    - `evaluate_tc_trial(tc)`：
      - 过滤出满足 `T < Tc` 和 `t_min <= T <= t_max` 的点
      - 计算 `x = log(Tc - T)`，`y = log(M)`，做线性回归
      - 计算 R²，打上 `is_valid` 标记
    - `find_best_tc`：
      - 从所有 `LogLogResult` 中选出 `is_valid` 且 R² 最大的一条，给出最佳 Tc 和 β

- 模块 `tc_trial.rs`
  - 提供 `generate_log_log_plot` 函数
    - 使用最佳 Tc 和 β 重新构造 `log(M)` vs `log(Tc - T)` 数据
    - 画出散点和拟合直线，生成 HTML 图像文件

### 使用方法的整体流程
1. 运行温度扫描，得到 \(T_i, M_i\) 数据并保存为 CSV。
2. 为 Tc 分析选择参数：
   - 分析区间 `[t_min, t_max]`（一般选在接近预计 Tc 的一小段温度范围内）
   - Tc 搜索区间 `[tc_min, tc_max]` 以及步长 `tc_step`
3. 运行 Tc 分析（log-log 方法二）：
   - 扫描候选 Tc
   - 对每个 Tc 做 `log(M)` vs `log(Tc - T)` 拟合
   - 选出 R² 最大的 Tc 与对应 β
4. 检查输出文件：
   - Tc 扫描结果 CSV
   - 详细结果 HTML（列出所有候选 Tc 的 β、R² 等）
   - log-log 拟合图（散点 + 直线），直观验证线性标度是否良好

