# Caco-2 Track A v1 脚本运行指南

本文说明本目录脚本的用途、依赖、运行顺序和产物。对应任务清单为项目根目录的 `endpoint_1_caco_trackA_baseline.md`；数据构建细节见 `endpoint_1_caco_trackA_dataset_build.md`。命令均从**项目根目录**运行，路径相对于项目根目录。`run_*.sh` 是记录命令、环境和退出状态的入口；同名 `.py` 是具体实现。跨版本的可复用代码在 `src/admet_pbpk/`。

## 1. 当前状态与使用方式

本工作区已经完成 16 个核心基线的五组验证、配置冻结、两套最终协议和 Final Benchmark Gate。结果在 `caco_trackA_v1/results/baselines/`；`final_benchmark_gate.json` 为 `PASS`。若只是查看或核对本次结果，直接读取已保存的矩阵、清单和运行日志即可，**不要从下载命令重新执行整条流水线**。

本目录很多写入脚本采用“一次写入”规则：目标存在时会拒绝覆盖，或仅在内容完全相同时接受。特别是 validation、最终模型、预测、矩阵与门禁报告不能在当前结果目录中简单重跑。若要完整重现实验，应使用独立的项目副本或新模型版本目录，保留当前 `raw/`、现有结果和 `progress/runs/`。不要为了让脚本成功而删除已归档的产物。当前工作区还有未提交文件，Git HEAD 不能单独代表完整运行代码；复现前应固定代码快照，并保留各运行日志里的脚本 SHA-256 和 Git 状态。

## 2. 环境与前置条件

已完成实验的主要软件版本来自冻结模型元数据：Python 3.11.16、PyTDC 1.1.15、RDKit 2023.09.6、NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.9.1、XGBoost 3.2.0。还需要脚本实际导入的依赖，例如 SciPy 和 joblib。运行时使用已准备的 `admetpbpk` 环境；若在新环境中复现，应先检查安装版本是否与冻结配置一致。这里没有经过验证的全量依赖锁文件，不能仅凭包名保证数值完全复现。

```bash
conda activate admetpbpk
cd /home/shahab/model-5
python -c "import sys, tdc, rdkit, numpy, pandas, sklearn, xgboost; print(sys.executable, rdkit.__version__, numpy.__version__, pandas.__version__, sklearn.__version__, xgboost.__version__)"
```

以上 `cd` 是本工作站路径；在独立副本中替换为该副本的项目根目录。下载官方数据需要访问 TDC；后续步骤读取本地归档。正式 XGBoost 验证与最终训练固定使用 `device=cuda`、`tree_method=hist`，需要可见的 NVIDIA GPU。脚本会检查实际训练设备；仅设置参数为 CUDA 而实际退回 CPU 不算成功。Ridge、SVR、随机森林、ExtraTrees 和 Null 基线按冻结实现运行。CPU 数组传给 GPU XGBoost 预测时可能出现 DMatrix 回退警告，这与训练设备检查是两回事。

所有实验使用原始 TDC `Y` 尺度，计算平均绝对误差（mean absolute error，MAE）；不要对 `Y` 再取对数。官方数据为 728 行 `train_val`、182 行固定 test；五个官方训练/验证划分各为 637/91 行。正式训练与调参不能读取固定 test 的 `Y`。只有完成全部盲预测审计后，独立评估入口才读取 test 标签。

## 3. 从新副本开始的执行顺序

下列是**新实验副本**的完整顺序，不是让当前已完成的目录再次执行。每一步成功后检查其 `progress/runs/<run_id>/exit_status.txt` 为 `0`，再进入下一步。脚本产生的源文件和模型产物应继续保留。

### 3.1 官方数据与行身份

```bash
bash caco_trackA_v1/scripts/run_download_tdc.sh
bash caco_trackA_v1/scripts/run_prepare_splits.sh
bash caco_trackA_v1/scripts/run_qc_dataset.sh
bash caco_trackA_v1/scripts/run_audit_baseline_archive.sh
bash caco_trackA_v1/scripts/run_audit_baseline_splits.sh
bash caco_trackA_v1/scripts/run_audit_baseline_test_boundary.sh
```

`download_tdc.py` 通过 PyTDC `admet_group` 的 BenchmarkGroup 获取 `Caco2_Wang`，保存原始文件和 `data/caco_trackA/raw/dataset_manifest.json`；不以单任务加载器自行重划分 test。`prepare_splits.py` 保存 `row_id` 和五组官方划分。`qc_dataset.py` 记录结构、重复和划分检查。后三个审计入口核对官方归档、划分行序和固定 test 的文件边界；数据审计不能用 test 标签选择模型方法。

### 3.2 分子表征与特征检查

```bash
bash caco_trackA_v1/scripts/run_build_morgan_features.sh
bash caco_trackA_v1/scripts/run_build_physchem10.sh
bash caco_trackA_v1/scripts/run_freeze_rdkit2d_descriptors.sh
bash caco_trackA_v1/scripts/run_build_rdkit2d.sh
bash caco_trackA_v1/scripts/run_validate_features.sh
bash caco_trackA_v1/scripts/run_audit_morgan_leakage.sh
```

Morgan 指纹、PhysChem-10 和 RDKit2D 的矩阵及 `features_manifest.json` 均放在 `data/caco_trackA/process/features/<representation>/`。RDKit2D 必须先冻结名称及顺序，再构建矩阵。`validate_features.py` 核对行数、行序、维度、缺失值、非有限值，以及标签 `Y` 和行标识没有进入特征矩阵。RDKit2D 中预先记录的少量缺失值在后续 pipeline 的训练部分填补，不能凭 test 表现删行或选填补方法。

`run_check_validation_only_flow.sh` 是先前的只训练/验证小规模流程检查；`run_audit_baseline_preprocessing.sh` 检查拟合型预处理只在训练部分拟合。它们不产生正式测试成绩，可在正式验证前执行：

```bash
bash caco_trackA_v1/scripts/run_check_validation_only_flow.sh
bash caco_trackA_v1/scripts/run_audit_baseline_preprocessing.sh
```

### 3.3 候选、实验配置和验证

```bash
bash caco_trackA_v1/scripts/run_freeze_baseline_candidates.sh
bash caco_trackA_v1/scripts/run_freeze_baseline_experiments.sh
bash caco_trackA_v1/scripts/run_record_baseline_inventory.sh
bash caco_trackA_v1/scripts/run_baseline_validation.sh --audit-inputs
```

候选文件在 `configs/baselines/search_space/`，实验配置在 `configs/baselines/`。`--audit-inputs` 只检查候选、728 行训练池、五组划分和输入边界，不训练模型。`run_record_baseline_inventory.sh` 保存当次数据校验值、软件版本、Git HEAD/未提交状态与随机种子。若要在 GPU 工作站预检查 XGBoost，可运行 `audit_xgb_compatibility.py --output <新报告路径>`；没有 GPU 时只能加 `--skip-fit` 检查参数兼容性，这**不能**代替正式 CUDA 拟合检查。

验证阶段按 `configs/baselines/experiments.json` 的冻结顺序，对下列 16 个实验分别执行一次：

```text
B00_NULL_MEAN       B01_NULL_MEDIAN
B10_PC10_RIDGE      B11_PC10_SVR       B12_PC10_RF       B13_PC10_ET       B14_PC10_XGB
B20_RDKIT2D_RIDGE   B21_RDKIT2D_SVR    B22_RDKIT2D_RF    B23_RDKIT2D_ET    B24_RDKIT2D_XGB
B30_MORGAN_SVR      B31_MORGAN_RF      B32_MORGAN_ET     B33_MORGAN_XGB
```

例如：

```bash
bash caco_trackA_v1/scripts/run_baseline_validation.sh --experiment B10_PC10_RIDGE
bash caco_trackA_v1/scripts/run_baseline_validation.sh --experiment B14_PC10_XGB
```

若在能访问 GPU 的工作站从头运行全部 16 项，也可以按冻结配置文件的原始顺序逐项调用；这段循环会写入正式验证结果，**只用于新的实验副本**：

```bash
mapfile -t experiments < <(python -c 'import json; print("\n".join(e["experiment_id"] for e in json.load(open("configs/baselines/experiments.json"))["experiments"]))')
for exp in "${experiments[@]}"; do
    bash caco_trackA_v1/scripts/run_baseline_validation.sh --experiment "$exp" || exit 1
done
```

每项使用同一批冻结候选，在 `split_seed=1..5` 的官方划分上评估；每组只用对应 train 拟合填补、方差筛选、标准化和模型，valid 用于比较配置。XGBoost 三项必须在可见 GPU 的环境中运行。验证只从训练池读取标签，不打开固定 test。输出位于 `caco_trackA_v1/results/baselines/<experiment_id>/validation/`，含 `candidate_results.csv`、`best_config.json`、`validation_summary.json`。完整 16 项到齐后执行：

```bash
bash caco_trackA_v1/scripts/run_audit_baseline_validation_results.sh
bash caco_trackA_v1/scripts/run_summarize_baselines.sh --check-readiness
bash caco_trackA_v1/scripts/run_summarize_baselines.sh --write-validation-matrix
bash caco_trackA_v1/scripts/run_freeze_baseline_configs.sh
```

`--check-readiness` 仅报告缺项；`--write-validation-matrix` 写出 16 行 `baseline_validation_matrix.csv`。配置冻结入口核对全部验证结果后生成各实验的 `frozen_config.json` 和 `validation_gate.json`。必须确认 Validation Gate 为 `PASS`，才能进行最终测试。配置选取基于未四舍五入的平均验证 MAE，不能在看到固定 test 误差后改动。

### 3.4 两套最终协议

每个实验都运行两套协议。`tdc_compatible_5split` 按官方 `split_seed=1..5` 分别使用 637 行训练，再分别对**同一固定 test 182 行**预测。`full_train_refit` 用完整 728 行 `train_val` 重新拟合；RF、ET、XGBoost 各用 `model_seed=1..5` 训练五次，Null、Ridge、SVR 各运行一次。两套协议的样本标准差含义不同，不能混在一起；单次重训的 SD 为 `N/A`。

对 16 个实验各执行下面两条命令。XGBoost 三项需要 GPU；其余按冻结实现运行。当前脚本每个协议内会按**全部训练 → 全部无标签预测 → 盲审计 → 独立评估 → 结果审计**执行。

```bash
bash caco_trackA_v1/scripts/run_baseline_final.sh --experiment B10_PC10_RIDGE --protocol tdc_compatible_5split --stage all
bash caco_trackA_v1/scripts/run_baseline_final.sh --experiment B10_PC10_RIDGE --protocol full_train_refit --stage all
```

将示例中的实验编号依次替换为上节列出的 16 项。若某个协议必须分阶段恢复，可用 `--stage train`、`predict`、`audit-blind`、`evaluate`、`audit-evaluated`，保持这个顺序；`evaluate` 会再次确认全部盲预测已通过审计。不要绕过盲审计单独读取 test `Y`，也不要用测试成绩修改冻结配置。`run_train_full.sh --audit-inputs` 可在正式训练前只读检查训练输入，不生成模型。

在新的副本中、所有 16 个配置已经冻结且 GPU 可见时，两套协议可按冻结顺序执行：

```bash
mapfile -t experiments < <(python -c 'import json; print("\n".join(e["experiment_id"] for e in json.load(open("configs/baselines/experiments.json"))["experiments"]))')
for exp in "${experiments[@]}"; do
    for protocol in tdc_compatible_5split full_train_refit; do
        bash caco_trackA_v1/scripts/run_baseline_final.sh --experiment "$exp" --protocol "$protocol" --stage all || exit 1
    done
done
```

每次运行的目录为 `caco_trackA_v1/results/baselines/<experiment_id>/final/<protocol>/<run_name>/`。其中 `model_metadata.json` 记录版本、输入和配置 SHA-256、种子、训练行数、Git 状态与模型产物；非 Null 模型保存 `model.joblib`，Null 基线记录训练集常数及不保存模型文件的原因。`blind_predictions.csv` 不含真实标签；独立评估后才生成 `evaluated_predictions.csv`、`metrics.json`。每个协议还保存 `blind_audit.json`、`evaluation_audit.json` 和 `summary.json`。

### 3.5 实验登记、最终矩阵和门禁

在生成最终矩阵前，`progress/experiments.csv` 必须有 **16 × 2 = 32 条**唯一的 `(experiment_id, benchmark_protocol)` 记录，且每条均指向相应 `summary.json` 与 `evaluation_audit.json`。本目录的最终汇总脚本**不会自动填写这张登记表**。新副本应从已通过审计的 `summary.json` 逐条登记 `run_count`、`MAE_mean`、`MAE_SD`（单次为 `N/A`）、SD 定义、标签尺度、Validation Gate SHA-256 和结果路径；列结构如下，可参照当前 `progress/experiments.csv`：

```text
dataset,model_version,experiment_id,benchmark_protocol,status,test_rows_per_run,run_count,primary_metric,MAE_mean,MAE_SD,MAE_SD_definition,label_scale,validation_gate_sha256,summary_path,evaluation_audit_path
```

其中 `status` 为 `evaluated_and_audited`、`primary_metric` 为 `MAE`、`test_rows_per_run` 为 `182`；`validation_gate_sha256` 是 `validation_gate.json` 的文件 SHA-256。`summary_path` 和 `evaluation_audit_path` 使用项目根目录相对路径。登记只能转录已经审计通过的结果，不用于重新挑选模型。

登记完成后运行：

```bash
bash caco_trackA_v1/scripts/run_summarize_final_baselines.sh
bash caco_trackA_v1/scripts/run_audit_final_benchmark_gate.sh
```

前一入口写出 32 行 `baseline_final_matrix.csv` 和包含来源校验值的 manifest；它要求所有审计、登记与冻结配置一致。后一入口复核 16 个实验、32 组协议、逐次模型和预测文件、官方行身份、日志、指标以及最终矩阵，并写出 `final_benchmark_gate.json`。只有门禁为 `PASS`，才能在任务清单中标记 `CORE BASELINE MATRIX = COMPLETE`。门禁检查的是核心基线的可复现性和评估边界，不代表药物的 PBPK 模型已验证。

## 4. 其他审计与独立工具

| 脚本 | 用途与调用时机 |
|---|---|
| `audit_label_scale.py --si <SI 工作簿路径>` | 用已核实的 Wang 补充材料和训练池记录交叉核对少量原始标签；依赖外部 SI 文件与预先记录的 SHA-256，不是完整流水线的必需运行入口。 |
| `audit_prediction_boundary.py` | 静态审阅预测/评估代码的数据读取边界；保留在已有审计记录中。 |
| `audit_xgb_compatibility.py --output <新路径> [--skip-fit]` | 检查冻结 XGBoost 参数组合；不加 `--skip-fit` 时在 GPU 上做代表性拟合。 |
| `audit_predictions.py` | 由 `run_baseline_final.sh` 调用；分别执行 `--mode blind` 与 `--mode evaluated`。 |
| `audit_final_benchmark_gate.py` | 由对应 `run_*.sh` 调用；已有门禁报告时拒绝覆盖。 |

运行日志统一在项目根目录 `progress/runs/`，通常包含 `command.txt`、`environment.txt`、起止时间、`stdout.log`、`stderr.log` 和 `exit_status.txt`。正式验证/最终训练日志还记录 Git 状态、主机和输入或配置 SHA-256。先看退出码，再看审计 JSON 的 `status`；不要只依据终端最后一行判断实验成功。

## 5. 常见中断与边界

- **提示输出已存在**：这是防止覆盖的预期保护。先检查原产物与日志；若确需新实验，使用独立副本或新的模型版本，不要删除当前结果。
- **XGBoost 找不到 CUDA 或退回 CPU**：正式 XGBoost 运行失败；检查工作站 GPU 可见性与环境。不要把 CPU 结果冒充本版已冻结的 GPU 协议。
- **配置或文件 SHA-256 不匹配**：停止运行，核查是否混用了不同版本的数据、代码、依赖或候选文件；不要为通过门禁而改写旧清单。
- **固定 test 或逐次预测不是 182 行**：本次完整 benchmark 评估未完成。保留失败记录，查明行身份或特征处理问题。
- **盲审计未通过**：不能进入读取 test `Y` 的独立评估阶段。
- **结果不等于当前矩阵**：先核对软件版本、数据校验值、官方划分、代码快照、随机种子和实际 CUDA 设备。不能按测试误差重新选模型或调参。

本指南记录的是 `caco_trackA_v1` 的核心基线流程；其他端点或模型版本必须使用各自的配置、数据目录、入口和结果目录。
