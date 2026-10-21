# O3 Gate 4B: strict directional refinement

Gate 4A 已经证明 O3 的 SA2/SA3 目标两态流形高度良态，并在反对称伸缩方向得到可重复的 mixed nuclear--variational nonclosure。Gate 4B 用更严格的波函数收敛和四级正负双向 refinement 判断该信号属于平滑 mixed curvature 还是局域 variational discontinuity。

## 计算规模

- 固定 `r1 + r2 = 2.7288 Angstrom`；中心角 `119.70 degree`；
- 强制所有几何使用 Cs；
- SA2 与 SA3，CAS(18,12)/cc-pVTZ；
- 反对称伸缩步长：`0.010, 0.005, 0.0025, 0.00125 Angstrom`；
- 角度步长：`0.20, 0.10, 0.05, 0.025 degree`；
- 每一级均包括正、负两个方向；
- 17 个唯一几何，34 个 SA-CASSCF。

Gate 4A 检查点只用于初始轨道。Gate 4B 会在严格阈值下重新优化中心和全部邻点，不会把 Gate 4A 的较松收敛结果直接混入 scaling：

```text
RHF energy tolerance                    1.0e-12
FCI tolerance                           1.0e-12
CASSCF energy tolerance                 1.0e-10
PySCF CASSCF orbital-gradient target    1.0e-5
Maximum CASSCF macro iterations         60
Explicit stationarity window            5 macro iterations
Stationarity energy-change ceiling      1.0e-10 Eh
Stationarity orbital-gradient ceiling   1.0e-4
Stationarity density-change ceiling     1.0e-7
Stationarity orbital-rotation ceiling   1.0e-7
```

全部 SA2/SA3 点统一使用 PySCF one-step SA-CASSCF，不在 production 数据中使用
实验性的 second-order/Newton CASSCF。PySCF 的 nominal 轨道梯度目标为 `1.0e-5`。
若其内部 `converged` 标志仍为 false，程序只在最后连续 5 个 macro iteration 同时
满足上述四个 stationarity ceilings 时接受该波函数，并将验收模式和最终数值写入
checkpoint。这个 ceiling 不是单独的梯度收敛标准：能量、密度和轨道旋转必须同时
证明迭代已经进入数值静止平台。`1.0e-4` 只是多指标保护上限，并非 production 的
常规梯度目标。对 `delta_r=-0.0025 A` 的 SA2 试算，最后 5 个 macro iteration 的
最大能量变化为 `2.84e-14 Eh`，最大密度变化为 `1.65e-9`，最大轨道旋转为 0，
而报告的轨道残差锁在 `5.09e-5`。该点只有在这些条件连续同时成立时才按
`stationary_floor` 验收；所有验收数值均写入 checkpoint，供后续审计。Newton 只
保留为可选诊断，不参与最终 refinement 数据。

几何按每个方向由粗到细排列。除每条正、负方向 continuation chain 的第一个点使用
protocol-matched center 轨道外，其余点均使用同方向、同协议的上一层较粗网格波函数
作为初始轨道。断点续跑时，已有 checkpoint 也会重新载入 continuation chain。
这既保持 SA2/SA3 的协议独立性，又避免在最细网格上反复从 center 作大跨度投影。
最细网格的 SA3 点可能在 15 个 macro iterations 后仍处于正常下降阶段，因此允许
最多 60 个 macro iterations；显式 stationarity test 仍仅在所有四项指标同时达到
原有阈值时验收，迭代预算的增加不会放松任何收敛判据。

## 运行

解压 Gate 4B 包，并把 Gate 4A 的工作目录位置告诉脚本：

```bash
tar -xzf O3_gate4B_inputs_v5_2.tar.gz
cd O3_gate4B
conda activate pyscf
O3_GATE4A_WORK=/mnt/O3/O3_gate4A/O3_gate4A_work bash run_O3_gate4B.sh
```

如果 Gate 4B 文件夹与 `O3_gate4A_work` 相邻，脚本会自动找到种子目录，可以直接：

```bash
bash run_O3_gate4B.sh
```

默认使用 32 核和 120000 MB 内存。调整方法：

```bash
O3_THREADS=16 O3_MEMORY_MB=60000 \
O3_GATE4A_WORK=/path/to/O3_gate4A_work \
bash run_O3_gate4B.sh
```

只检查几何和任务数而不导入 PySCF：

```bash
python O3_gate4B_run.py \
  --seed-workdir /path/to/O3_gate4A_work \
  --dry-run
```

Slurm 用户修改 `submit_O3_gate4B.slurm` 中的分区、账户和环境初始化后提交：

```bash
sbatch submit_O3_gate4B.slurm
```

默认结果写入新的 `O3_gate4B_work_v5`，从而避免混入此前试算产生的
checkpoint。计算可断点续跑；只有参数、几何、阈值和 optimizer 均匹配的 Gate 4B
检查点才会被跳过。

## 自动分析

脚本在全部计算完成后自动：

1. 用严格多电子交叉重叠提取每个 SA3 中与 SA2 最匹配的 m=2 子空间；
2. 计算 16 个 polar-unitary plaquette 的 `C_qp`、`eta_edge` 和 edge leakage；
3. 对正向、负向、maximum、mean 和 RMS 分别拟合 `C_qp proportional to h^alpha`；
4. 执行独立随机酉规范变换自检；
5. 若环境中有 matplotlib，输出无网格线、ticks-in、仅 major ticks 的 600 dpi PNG/TIFF 以及 PDF。

主要结果文件：

- `O3_gate4B_work_v5/O3_gate4B_plaquettes.csv`
- `O3_gate4B_work_v5/O3_gate4B_scaling.csv`
- `O3_gate4B_work_v5/O3_gate4B_fits.csv`
- `O3_gate4B_work_v5/O3_gate4B_subspaces.csv`
- `O3_gate4B_work_v5/O3_gate4B_summary.json`
- `O3_gate4B_work_v5/O3_gate4B_refinement.png`
- `O3_gate4B_work_v5/O3_gate4B_refinement.tiff`
- `O3_gate4B_work_v5/O3_gate4B_refinement.pdf`

## 判据

- `alpha approximately 1`：固定协议变化下的平滑 mixed curvature；
- `alpha approximately 0`：refinement-resistant variational discontinuity；
- 弯曲方向无稳定幂律且进入收敛底噪：该方向近似对易；
- `eta_edge` 必须在全部 refinement 中保持良态，否则不能单独解释 `C_qp`；
- `gauge_test_abs_error` 应接近机器精度。

跑完后最好将整个 `O3_gate4B_work_v5` 目录连同三个 stdout 文件一起打包发回。
