LiH aug-cc-pVTZ 一键完整计算包
版本：v1.0 / 2026-10-13

这是 JCTC 多态闭环输运返修的 LiH R–Fz 算例。
独立运行，不需要 S0/S1/S2/S3 旧脚本、旧工作目录或旧波函数。
唯一计算主程序为 LiH_augccpVTZ_one_shot.py；其他文件提供运行说明和测试记录。

一、直接运行（Linux，已激活安装 PySCF 的 Python 环境）

  tar -xzf LiH_augccpVTZ_one_shot.tar.gz
  cd LiH_augccpVTZ_one_shot
  nohup bash run_all.sh 8 > run.log 2>&1 &

8 表示每个计算使用的线程数，可按实际分配的 CPU 数修改。
这是顺序计算各个点、每点多线程；不是同时启动 8 个任务。
默认内存预算 8000 MB，不是硬性内存上限。修改示例：

  LIH_MEMORY_MB=16000 nohup bash run_all.sh 16 > run.log 2>&1 &

监视进度：
  tail -f run.log

缺少依赖时，先在自己的计算环境运行：
  python -m pip install -r requirements.txt

实际验证环境：Python 3.12 / PySCF 2.13.1 / NumPy 2.3.5 / SciPy 1.17.0。
不自动安装软件，不修改原有计算环境。

二、一次自动执行的内容

1. 数学自检：完整四电子行列式重叠与 PySCF 的全行列式空间重叠交叉验证；
   U(4) 规范协变性及两路线/闭环范数一致性。
2. 原 S3 固定窗口 4×5 网格：R=5.80,5.85,5.90,5.95 Å；
   Fz=0.01050,0.01075,0.01100,0.01125,0.01150 a.u.。
3. 同一外回路，每边 1、2、4、8、16 段（共有 4、8、16、32、64 条边）。
4. 以 (R,Fz)=(5.875 Å,0.011 a.u.) 为中心，边长分别缩小为
   1/2、1/4、1/8、1/16；每个缩小回路分别用每边 4 和 8 段。
5. 最细外回路的正向和反向历史：各 65 次优化，起点和独立重优化的闭合点均保留。
   每点用前一点的轨道投影初始化，并与独立 RHF 初猜路线比较。
6. 完整重叠、极分解输运、相同端点两路线比较、Wilson 本征相位、偶极矩阵回输，
   驻点/自旋/轨道正交性/第五根隔离性/边保真度/态指派/历史一致性检查。
7. JSON 与 CSV 汇总，并打包所有原始 npz 波函数、逐点日志、程序和结论。

去重后共 202 个独立初猜点（每点固定执行两条优化路线）+ 130 个历史点，
共 534 个优化任务；未收敛时还可能有自动 Newton 重启。
运行时间取决于机器、线程设置和收敛；程序会逐点打印时间与进度。
不扫描或另选“信号最大”的窗口，不以预期的 C 值替代计算结果。
这个包只计算 aug-cc-pVTZ；不会重算另两个基组，也不要求它们的结果。

三、固定电子结构模型

- 中性 LiH，Li 在 z=-R/2，H 在 z=+R/2；坐标用 Å，电场用原子单位。
- aug-cc-pVTZ，球谐高斯基组；C2v。
- 等权 singlet-A1 SA4-CASSCF(2e,5o)，四个根的权重均为 1/4。
- 一个双占据 inactive core，与其余轨道一起优化；没有 frozen orbital。
- 活性轨道表示组成 A1^3 B1^1 B2^1。
- 保留原 S3 的 occupied-valence-locked target-AO 初猜选择器，
  避免弥散虚轨道挤掉唯一的占据价轨道。
- H(Fz)=H(0)-mu_z Fz；电子单电子项 +Fz*r_z，核项 -Fz*sum(Z_A R_A,z)。
- 外部隔离性仅由同一轨道下的第五个 singlet-A1 CASCI 根探测；
  它不证明相对于所有其他空间/自旋对称态的完整隔离性。

与归档求解器相比，本包收紧了数值精度：
RHF 能量 1e-13 Eh；CASSCF 能量 1e-12 Eh；轨道梯度目标 1e-8；
FCI 1e-13；AH conv_tol=1e-16、lindep=1e-20。
收紧 AH 阈值是为了让较小回路的梯度目标实际可达到，物理模型不变。
每个独立点均执行两条预先固定的路线：
A. one-step 优化（必要时 Newton 重启）；B. 从同一个固定 RHF 初猜直接 Newton。
在通过收敛、驻点、自旋、正交性和 CASCI 一致性检查的候选中，选 SA4 平均能量最低者。
两套原始波函数与日志都保存在 candidates/，选取依据保存在每个选中点的 metadata 中。
选择不使用闭环信号、边保真度或希望得到的答案，也不保证找到了全局最低驻点。
这是对原归档“每点单条独立优化路线”的明确改进：交付前测试发现
(5.80 Å,0.0105 a.u.) 的 one-step 路线能收敛到较高的不同驻点，
直接 Newton 得到更低的同一 SA4 目标值。固定双路线用于检测并减少这类风险。
历史路线仍从前一点投影初始化，独立检验这种选择是否可重复。
各路线主优化未收敛时，最多作三次 Newton 重启。
没有把未收敛结果强行当作收敛结果。
最终驻点检查要求最大轨道梯度 <=1e-6，比原归档的 1e-4 更严格。

四、重叠与输出定义

完整四电子重叠包括 inactive core 和 core-active 交叉项。
记前六个（1 core + 5 active）MO 的跨几何重叠为 X，
D_ac = X_00 X_ac - X_a0 X_0c，a,c 为 active 轨道。
M_ij = sum_abcd C_i,ab* C_j,cd D_ac D_bd。
程序不除以 core overlap，也不以 active-only overlap 代替完整重叠。

边输运 T 是 M 的 unitary polar factor。K_RF/K_FR 把终点态坐标带回起点坐标系。
C = ||K_RF-K_FR||_F / (2 sqrt(4)) = ||K_RF-K_FR||_F/4。
eta = 全部边最小奇异值中的最小值；leakage = max_edge[1-mean(sigma^2)]。
最优指派最大化 sum_i |M_i,p(i)|^2；margin 为最优与次优差值除以 4。
D_mu = 三个偶极分量回输差异的联合 Frobenius 范数 / 终点偶极矩阵联合范数。
偶极回输差异是同一终点算符在共同起点框架下的表示差异，不能称为终点物理偶极改变。

缩小回路面积采用 Å × 电场原子单位；报告 4C/area 的局部曲率范数估计，
单位为 (Å × a.u. field)^(-1)。请结合 4/8 分段差异和 log(C)-log(area) 斜率判断。

主回路边保真度阈值 eta>=0.98、leakage<=0.01；指派 margin>=0.10。
第五根 gap>=0.01 Eh；自旋误差<=1e-7；轨道正交误差<=1e-9；
五根探测与四个优化根的能量差<=1e-8 Eh。
1、2 分段回路只是粗网格诊断，保真度不合格不应阻止细网格计算。

最细外回路和全部缩小回路需要合格；外回路 8→16 及缩小回路 4→8 的
C、D_mu 差值需 <=max(1e-8, 0.001×细网格结果)。这是本包明确记录的新数值判据。
正反历史沿用归档的严格量级要求：平均能量差<=1e-8 Eh、
子空间 eta>=0.9999、leakage<=1e-6，活性轨道子空间同样检查；
|delta C|、|delta D_mu| 均<=1e-8。
相同根指派、信号大小和面积标度的实际结果均单独报告，不预设它们一定成立。

五、结果在哪儿

默认目录：LiH_augccpVTZ_results/
- SUMMARY.txt：首先查看。
- LiH_augccpVTZ_decision.json：QUALIFIED 或 REVIEW_REQUIRED，以及逐项判据。
- loop_summary.csv：主回路、缩小回路的 C、D_mu、保真度和曲率估计。
- point_summary.csv：独立初猜点的能量、梯度和隔离 gap。
- loop_results.json：完整边重叠、输运矩阵和各项自检。
- history_results.json：正反路径、逐点对比和独立闭合点对比。
- candidates/：每个独立点两条优化路线的完整结果。
- solutions/、histories/：每一点的原始 MO/CI/dipole npz 和 PySCF 日志。
- plan.json：开始计算前保存的完整协议、版本、源码 hash。
- wavefunction_sha256.json：全部波函数 hash。
- run_status.json：运行/完成/失败状态。

run_all.sh 完成后自动生成 LiH_augccpVTZ_results.tar.gz。
把这个压缩包发回即可进一步检查结果和更新正文/SI。
退出码 0：完成且数值判据通过；2：完成但需要审查，仍会打包；
其他非零值：异常或波函数检查失败，详见日志。
QUALIFIED 是这些固定数值判据的结果，不是对论文结论或发表的保证。

六、断点续算和其他命令

意外中断：再次运行同一条 run_all.sh 命令，会跳过已完成且协议一致的点。
历史检查点还核对前驱波函数 hash；不混用来自其他历史/程序版本的结果。
同一结果目录禁止同时运行两个 all/analyze 进程。
若程序、Python 科学库版本或计算协议不同，使用新目录，例如：

  LIH_ROOT="$PWD/LiH_augccpVTZ_results_new" bash run_all.sh 8

不自动覆盖未收敛或不兼容的旧波函数；发生这种情况时保留日志供判断原因。
重新分析已有完整结果（无需重新优化）：

  python LiH_augccpVTZ_one_shot.py analyze --threads 8

仅打印完整方案：
  python LiH_augccpVTZ_one_shot.py plan

仅数学自检：
  python LiH_augccpVTZ_one_shot.py selftest --threads 1

四点实际计算测试：
  python LiH_augccpVTZ_one_shot.py smoke --threads 2 --root smoke_test

smoke 是独立验证目录，不混入正式计算。默认 all 已自动执行数学自检。

七、来源与验证范围

计算函数来源：原 S0/S1 电子结构求解器、S3 价轨道锁定初猜选择器，
以及 2026-10-13 返修包的完整四电子重叠实现。原文件 SHA256 嵌在程序中。
PySCF 官方方法文档：https://pyscf.org/user/mcscf.html
官方 state-average 实现：https://pyscf.org/_modules/pyscf/mcscf/addons.html

本包交付前做了数学自检、实际 aug-cc-pVTZ 点计算、四角闭环分析、
轨道投影历史与检查点续算测试；详细证据见 validation/。
完整的 534 个优化任务尚未在此测试环境跑完，需要在服务器运行上面的一键命令。
交付测试不代替服务器完整计算的科学判据。
