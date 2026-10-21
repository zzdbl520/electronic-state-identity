# O3 Gate 4A input package

这个包先验证 O3 是否值得进入 Figure 4 的完整二维扫描。它只计算 5 个几何点、两种变分协议，共 10 个 SA-CASSCF：

| 几何点 | delta r (Angstrom) | theta (degree) |
|---|---:|---:|
| center | 0.000 | 119.70 |
| dr_minus | -0.010 | 119.70 |
| dr_plus | +0.010 | 119.70 |
| theta_minus | 0.000 | 119.50 |
| theta_plus | 0.000 | 119.90 |

所有点固定 `r1 + r2 = 2.7288 Angstrom`，使用 CAS(18,12)/cc-pVTZ。协议为同权 SA2 与同权 SA3；二者分别优化最低 2 个和最低 3 个单重态 A-double-prime 根。

## 为什么强制 Cs

中心几何本来具有 C2v 对称性，PySCF 若自动识别会把两个目标态写成 A2 和 B1，而离开中心后它们都属于 Cs 的 A-double-prime。脚本在所有点显式设置 `mol.symmetry = "Cs"`，因此协议、根数和态标签在每条边上完全一致，不需要跨点群拼接。

## 直接运行

先激活已有 PySCF 环境，然后：

```bash
tar -xzf O3_gate4A_inputs.tar.gz
cd O3_gate4A
conda activate pyscf
bash run_O3_gate4A.sh
```

默认使用 32 核、120000 MB 内存。若机器配置不同：

```bash
O3_THREADS=16 O3_MEMORY_MB=60000 bash run_O3_gate4A.sh
```

脚本支持断点续跑；已经存在且参数匹配的波函数检查点会自动跳过。若使用 Slurm，先按集群规则修改 `submit_O3_gate4A.slurm` 中的账户、分区或环境初始化行，再运行：

```bash
sbatch submit_O3_gate4A.slurm
```

计算前可做不导入 PySCF 的几何检查：

```bash
python O3_gate4A_run.py --dry-run
```

## 自动分析

`run_O3_gate4A.sh` 在 10 个计算完成后自动调用分析程序。分析程序在每个几何点用 2x3 的同几何 SA2--SA3 多电子重叠矩阵做 SVD，以其前两个右奇异向量定义 SA3 内与 SA2 最匹配的 m=2 子空间。随后计算四个小方格：

- dr_minus -> center
- center -> dr_plus
- theta_minus -> center
- center -> theta_plus

每个方格中，`A = SA2(q0)`、`B = SA2(q1)`、`C = selected-SA3(q0)`、`D = selected-SA3(q1)`。

多电子交叉重叠不是 CI 系数点积。对 core/active 分块的轨道交叉重叠，程序使用精确的 Schur-complement 形式：

```text
M_IJ = det(S_cc)^2 <CI_I^X | CI_J^Y>_(S_eff)
S_eff = S_aa - S_ac S_cc^(-1) S_ca
```

因此 inactive-core 因子和 core--active 交叉项均被保留。每条边先取 overlap matrix 的 polar-unitary factor，再形成两条路径并计算 mixed nuclear--variational nonclosure。

## 跑完后请发回

最少发回以下 4 个文件；如有失败，再附对应日志：

1. `O3_gate4A.stdout`
2. `O3_gate4A_analysis.stdout`
3. `O3_gate4A_work/O3_gate4A_plaquettes.csv`
4. `O3_gate4A_work/O3_gate4A_subspaces.csv`

最好把整个 `O3_gate4A_work` 目录打包，因为 `wavefunctions/*.npz` 可直接用于后续 refinement 和 Figure 4，不需要重跑 Gate 4A。

关键检查量：

- `sigma_match_2`：SA2 与 SA3 最佳 m=2 子空间的第二奇异值；
- `SA3_gap12_Eh`：第三个 SA3 根与目标低能对之间的能量隔离；
- `SA3_root3_weight_in_selected_subspace`：第三个 SA3 根侵入目标子空间的程度；
- `eta_edge`：四条边中最小的最小奇异值；
- `C_qp`：polar-unitary mixed nonclosure；
- `gauge_test_abs_error`：随机独立酉变换后的不变量自检，正常应接近机器精度。

Gate 4A 的任务是判断数值条件和信号是否足以支持下一步定向 refinement；此时不预设 `C_qp` 必须大或必须小。
