# 现代优化器对比实验方案：MuonAdamW vs AdamW vs SOAP

> 基于 nanochat 代码库，在受限算力下对比三种前沿优化器在语言模型预训练中的表现。

## 1. 实验目标

在 nanochat 的现代化 GPT 架构上，系统性对比三种代表不同优化哲学的优化器：

| 优化器 | 核心思想 | 代表论文 |
|--------|---------|---------|
| **MuonAdamW** | 矩阵正交化：对二维权重执行 Polar Express 正交化，强制梯度能量均匀分散到各特征方向 | Muon (KellerJordan/modded-nanogpt) |
| **AdamW** | 自适应一阶矩估计：逐参数维护一阶/二阶动量，解耦权重衰减 | Loshchilov & Hutter, 2017 |
| **SOAP** | 预条件化二阶优化：在 Shampoo 预条件器的特征基中运行 Adam，周期性更新 Kronecker 因子化预条件矩阵 | Vyas et al., arXiv:2409.11321 |

**核心问题**：在相同的计算预算（FLOPs）和数据条件下，这三种优化器在收敛速度、最终模型质量、计算效率、显存占用上的定量差异如何？

---

## 2. 优化器技术对比

### 2.1 MuonAdamW（项目默认）

nanochat 已有实现（`nanochat/optim.py`），采用二元混合范式：

- **矩阵参数（Transformer 隐层权重）**→ Muon：SGD 动量 → Polar Express 正交化（5 步 Newton-Schulz 迭代）→ NorMuon 方差约减 → Cautious 权重衰减
- **非矩阵参数（嵌入层、标量）**→ AdamW：融合内核，指数移动平均一阶/二阶矩

**优化器状态**：矩阵部分需 1x 动量缓冲 + 因子化二阶动量（~1.01x 参数量），非矩阵部分需 2x（一阶+二阶矩）。

### 2.2 纯 AdamW

将现有 `adamw_step_fused` 内核应用于所有参数（包括矩阵参数），作为行业标准基线。

**优化器状态**：全参数 2x（exp_avg + exp_avg_sq）。

### 2.3 SOAP（新增实现）

SOAP = **S**hampo**O** with **A**dam in the **P**reconditioner's eigenbasis。核心洞见：Shampoo（1/2 幂次）等价于在其预条件器的特征基中运行 Adafactor。SOAP 的做法是：

1. **周期性**（每 `precondition_frequency` 步）对 Kronecker 因子化的预条件矩阵执行特征分解，得到旋转矩阵 Q
2. **每步**在旋转后的坐标系中执行标准 Adam 更新（维护旋转空间中的一阶/二阶矩）
3. 将更新旋转回原始参数空间

**优化器状态**：
- 旋转空间中的 Adam 状态：2x 参数量（一阶 + 二阶矩）
- 预条件矩阵 Q：对 (m, n) 形状参数，存储 Q_left (m×m) + Q_right (n×n)
- 总显存 ≈ 2x 参数量 + Σ(m² + n²)

**关键超参数**（相比 AdamW 仅多一个）：
- `precondition_frequency`：预条件器更新频率（推荐 10~16）

---

## 3. 实验架构

### 3.1 模型与数据

所有实验统一使用 nanochat 的 GPT 架构和 FineWeb 预训练数据集。

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 架构 | nanochat GPT | RoPE、QK-Norm、ReLU²、Value Embeddings、滑动窗口注意力 |
| 数据集 | FineWeb | nanochat 默认预训练语料 |
| 词表 | 32768 (BPE) | nanochat 默认词表 |
| 序列长度 | 2048 | `--max-seq-len=2048` |
| 精度 | bf16 (autocast) | 全程混合精度训练 |
| 随机性控制 | 全局种子固定 | `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` 固定权重初始化和数据加载顺序 |

### 3.2 实验规模

| 阶段 | 模型 | n_embd | 参数量 | 训练词元 | 单卡 Pro 6000 估时 |
|------|------|--------|--------|---------|-----------------|
| **超参校准** | d12 | 768 | ~124M | ~900M | 5-8 min/run |
| **主实验** | d20 | 1280 | ~430M | ~3.2B | 25-35 min/run |
| **消融实验** | d20 | 1280 | ~430M | ~3.2B | 25-35 min/run |

### 3.3 控制变量

以下条件在所有优化器对比中**严格固定**：

```
--depth=20
--max-seq-len=2048
--total-batch-size=524288      # 自动计算或固定
--target-param-data-ratio=10.5 # 确定训练词元总量
--warmup-ratio=0.05            # WSD 预热比例（见 3.4 说明）
--warmdown-ratio=0.5           # WSD 衰减比例
--final-lr-frac=0.0            # 最终学习率归零
--eval-every=250               # 评估频率
--seed=42                      # 全局随机种子
```

### 3.4 随机性控制与 WSD 预热策略

**随机种子**：在脚本入口严格固定 `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` + `random.seed(seed)` + `np.random.seed(seed)`，确保所有优化器面对完全相同的初始权重拓扑和数据流。不同优化器在最初几百步对初始权重曲率极其敏感，未固定种子将导致结果不可重复、不可比。

**WSD 预热比例**：基线统一设为 `warmup_ratio=0.05`（而非 0.0），原因如下：
- **SOAP** 在初始阶段需要积累足够的二阶统计量（Kronecker 因子），无 warmup 会导致初期特征分解严重偏离真实曲率，使 SOAP 在比较中处于不公正的劣势
- **MuonAdamW** 已内置动量 warmup（0.85→0.95），额外的 LR warmup 不会负面影响
- 5% 的 warmup 步数在 d12 约 90 步，开销极小但能有效消除冷启动偏差
- 在超参搜索阶段（阶段一）SOAP 额外搜索 `warmup_ratio ∈ {0.0, 0.05, 0.1}`

---

## 4. 超参数配置

### 4.1 超参数搜索空间

**阶段一**在 d8 上快速搜索每种优化器的最优超参数。

#### MuonAdamW（参考现有默认值微调）

| 参数 | 搜索范围 | 默认值 |
|------|---------|--------|
| `matrix_lr` | {0.01, 0.02, 0.03, 0.05} | 0.02 |
| `weight_decay` | {0.1, 0.2, 0.3} | 0.2 |
| `embedding_lr` | 固定 0.3 | 0.3 |
| `unembedding_lr` | 固定 0.004 | 0.004 |
| Muon 动量 | 固定 0.85→0.95 warmup | — |

#### 纯 AdamW

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `lr`（全局） | {1e-4, 2e-4, 3e-4, 5e-4, 8e-4} | 矩阵参数学习率 |
| `weight_decay` | {0.01, 0.05, 0.1} | 解耦权重衰减 |
| `beta1` | 固定 0.9 | — |
| `beta2` | {0.95, 0.99} | — |

#### SOAP

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `lr` | {1e-3, 3e-3, 5e-3, 1e-2} | SOAP 论文推荐 ~3e-3 |
| `beta1` | 固定 0.9 | 一阶矩衰减 |
| `beta2` | {0.95, 0.99} | 二阶矩衰减 |
| `precondition_frequency` | {10, 50} | 预条件器更新频率 |
| `weight_decay` | {0.0, 0.01} | — |
| `warmup_ratio` | {0.0, 0.05, 0.1} | SOAP 专属搜索维度（见 3.4） |

### 4.2 超参搜索策略

采用**分阶段顺序搜索**而非全网格，以控制 d12 校准的运行次数：

1. **主轴搜索**：先扫描 LR × WD/PF 的主要维度（使用默认 beta2/warmup）
2. **精细化搜索**：取主轴最优配置，再搜索次要维度（beta2、warmup、WD）

| 优化器 | 策略 | 运行数 |
|--------|------|--------|
| MuonAdamW | 4 LR × 3 WD 全网格 | 12 |
| AdamW | 5 LR × 3 WD → best 配 beta2 {0.95,0.99} | 15 + 2 = 17 |
| SOAP | 4 LR × 2 PF → best 配 WD/warmup/beta2 | 8 + 6 = 14 |
| **总计** | | **~43 次** |

- d12 上每次运行 ~500 步（约 5-8 分钟，`--device-batch-size=64`）
- 以 `val_bpb` 收敛曲线为依据选择最优配置
- 总搜索时间（单卡 Pro 6000）：**~4.5-5.5 小时**

### 4.3 跨尺度超参迁移校验（d12 → d20）

> **风险**：d12（124M 参数）上的最优学习率直接迁移到 d20（430M 参数）可能不准确甚至导致发散。nanochat 对 Muon 的 muP 缩放针对的是 depth 变化时的 LR 自动调整（`dmodel_lr_scale = (model_dim/768)^{-0.5}`），但其他优化器（AdamW、SOAP）未经过此缩放验证。

**迁移校验流程**（在阶段二正式对比之前执行）：

1. 取 d12 上每种优化器的最优 LR 值 `lr*`
2. 在 d20 上以 `lr ∈ {0.5×lr*, lr*, 2×lr*}` 各跑 **100 步**（~5 min/run）
3. 观察初期 train_loss 轨迹：是否平稳下降、是否出现爆炸或停滞
4. 选择 d20 上表现最稳定的 LR 作为正式对比的实际值
5. 仅需 ~9 次短运行（3 优化器 × 3 LR），额外耗时 **~45 min**

---

## 5. 评估指标体系

### 5.1 核心收敛质量指标

| 指标 | 定义 | 采集方式 | 频率 |
|------|------|---------|------|
| **val_bpb** | 验证集 bits-per-byte，词表大小无关的损失度量 | `evaluate_bpb()` on val split | 每 250 步 |
| **min_val_bpb** | 训练全程验证 bpb 最低值 | 跟踪 val_bpb 历史最小值 | 训练结束 |
| **final_val_bpb** | 最后一步的验证 bpb | 最终步评估 | 训练结束 |
| **CORE metric** | DCLM CORE 综合评分（涵盖多个下游任务） | `evaluate_core()` | 训练结束 |

### 5.2 训练效率指标

| 指标 | 定义 | 采集方式 | 频率 |
|------|------|---------|------|
| **steps_to_target** | 达到指定 val_bpb 阈值所需的训练步数 | 从 val_bpb 日志中提取 | 事后分析 |
| **time_to_target** | 达到指定 val_bpb 阈值所需的挂钟时间（秒） | `total_training_time` at target step | 事后分析 |
| **flops_to_target** | 达到指定 val_bpb 阈值所需的总 FLOPs | `total_training_flops` at target step | 事后分析 |
| **convergence_rate** | val_bpb 每千步的平均下降速率 | 线性拟合 val_bpb vs step | 事后分析 |

> **target 阈值设定**：以 AdamW 基线的 final_val_bpb 作为目标水位，计算其他优化器达到该水位的步数/时间/FLOPs。

### 5.3 系统工程指标

| 指标 | 定义 | 采集方式 | 频率 |
|------|------|---------|------|
| **tok_per_sec** | 每秒处理的 token 数（训练吞吐量） | `total_batch_size / dt` | 每步记录 |
| **avg_step_time** | 平均每步训练耗时（毫秒） | `dt` 的均值（排除前 10 步编译预热） | 每步记录 |
| **p99_step_time** | 第 99 百分位步耗时（毫秒） | 全程 `dt` 序列的 P99 分位数 | 事后分析 |
| **step_time_jitter** | 步耗时的变异系数 CV = σ(dt)/μ(dt) | 全程 `dt` 的标准差/均值 | 事后分析 |
| **bf16_mfu** | 模型 FLOPs 利用率（相对 GPU bf16 算力峰值） | `flops_per_sec / peak_flops` | 每步记录 |
| **peak_memory_mb** | GPU 峰值显存占用（MiB） | `torch.cuda.max_memory_allocated()` | 训练结束 |
| **optimizer_overhead** | 优化器步相对前向+后向的耗时占比 | 分段插桩计时 | 采样 100 步 |

> **SOAP P99 延迟预警**：SOAP 每 `precondition_frequency` 步执行一次特征分解（复杂度 O(d³)），d12 中 3072 维矩阵的特征分解显著慢于常规 Adam 更新。`p99_step_time` 和 `step_time_jitter` 是评估 SOAP 在分布式训练中落地可行性的关键指标——高 jitter 会导致严重的同步空泡（bubble）。建议在训练日志中绘制完整的步耗时分布直方图。

### 5.4 训练稳定性指标

| 指标 | 定义 | 采集方式 | 频率 |
|------|------|---------|------|
| **loss_variance** | 训练损失的滑动窗口方差 | 滑动窗口（100步）内 train_loss 方差 | 事后分析 |
| **loss_spikes** | 训练损失突变次数（>2x EMA 值） | 检测 train_loss 异常峰值 | 事后分析 |
| **grad_norm** | 全局梯度范数 | `torch.nn.utils.clip_grad_norm_` | 每步记录 |
| **nan_count** | 训练中出现 NaN 的次数 | 检测 loss 是否为 NaN | 每步检查 |

### 5.5 优化器微观动力学探针

| 指标 | 定义 | 采集方式 | 频率 |
|------|------|---------|------|
| **update_to_weight_ratio** | 逐层 ‖Δw‖/‖w‖，参数更新幅度与参数量级的比值 | 在 optimizer.step() 前后计算每个参数组的 `(w_new - w_old).norm() / w_old.norm()` | 每 100 步采样 |
| **per_layer_uwr** | 各层 update-to-weight ratio 的分布 | 按层分组记录到 wandb/日志 | 每 100 步采样 |

> **update-to-weight ratio 的意义**：仅监控 `grad_norm` 无法反映参数实际被更新的幅度，因为 Muon 的正交化和 SOAP 的预条件对梯度的缩放逻辑截然不同。当某优化器在某学习率下崩溃时，`‖Δw‖/‖w‖` 异常（通常 >0.1）往往是直接原因。健康范围一般在 1e-4 ~ 1e-2。

### 5.6 泛化质量探针

| 指标 | 定义 | 采集方式 | 频率 |
|------|------|---------|------|
| **train_bpb** | 训练集 bits-per-byte（当前 batch 的 loss 转换） | `train_loss * log2(e) / bytes_per_token` | 每步记录 |
| **generalization_gap** | val_bpb − train_bpb，泛化差距 | 在 eval 步同时计算 train_bpb 和 val_bpb 的差值 | 每 250 步 |

> **泛化差距的意义**：二阶优化器（SOAP）和伪二阶优化器（Muon）对训练集曲率的拟合更深，在有限数据下可能更容易过拟合。监控 `val_bpb - train_bpb` 能评估优化器是否只是在"记住"数据。若 SOAP 的 val_bpb 优于 AdamW 但 generalization_gap 也显著更大，则需要调整正则化策略。

### 5.7 评估指标汇总表

```
┌─────────────────────────────────────────────────────────────────┐
│                    评估指标五维矩阵（20 项）                    │
├───────────────┬─────────────────────────────────────────────────┤
│  收敛质量     │ val_bpb, min_val_bpb, CORE metric              │
├───────────────┼─────────────────────────────────────────────────┤
│  训练效率     │ steps/time/flops_to_target, convergence_rate   │
├───────────────┼─────────────────────────────────────────────────┤
│  系统工程     │ tok/sec, avg/p99_step_time, jitter, MFU,       │
│               │ peak_memory, optimizer_overhead                │
├───────────────┼─────────────────────────────────────────────────┤
│  训练稳定性   │ loss_variance, loss_spikes, grad_norm, NaN     │
├───────────────┼─────────────────────────────────────────────────┤
│  微观动力学   │ update_to_weight_ratio (逐层), train_bpb,      │
│  与泛化       │ generalization_gap                              │
└───────────────┴─────────────────────────────────────────────────┘
```

---

## 6. 实验流程

### 6.1 阶段一：d12 超参数校准（~5-7 hr）

在 d12 模型上快速搜索各优化器最优超参数。

```bash
# MuonAdamW 校准示例
python -m scripts.base_train --depth=12 --num-iterations=500 --eval-every=100 \
    --optimizer-type=muon --matrix-lr=0.02 --embedding-lr=0.3 \
    --run="sweep_muon_lr0.02" --core-metric-every=-1 --sample-every=-1 --save-every=-1

# 纯 AdamW 校准示例
python -m scripts.base_train --depth=12 --num-iterations=500 --eval-every=100 \
    --optimizer-type=adamw --matrix-lr=3e-4 \
    --run="sweep_adamw_lr3e-4" --core-metric-every=-1 --sample-every=-1 --save-every=-1

# SOAP 校准示例
python -m scripts.base_train --depth=12 --num-iterations=500 --eval-every=100 \
    --optimizer-type=soap --soap-lr=3e-3 --precondition-frequency=10 \
    --run="sweep_soap_lr3e-3_pf10" --core-metric-every=-1 --sample-every=-1 --save-every=-1
```

**选择标准**：以 val_bpb @step500 最低值确定各优化器的最优超参组合。

### 6.2 阶段二：d20 正式对比（~1.5-2 hr）

使用校准后的最优超参数在 d20 上进行全程训练。

```bash
# 1. MuonAdamW（基线）
python -m scripts.base_train --depth=20 --optimizer-type=muon \
    --matrix-lr=<calibrated> --weight-decay=<calibrated> --embedding-lr=0.3 \
    --run="cmp_muon" --model-tag="cmp_muon"

# 2. 纯 AdamW
python -m scripts.base_train --depth=20 --optimizer-type=adamw \
    --matrix-lr=<calibrated> --weight-decay=<calibrated> \
    --run="cmp_adamw" --model-tag="cmp_adamw"

# 3. SOAP
python -m scripts.base_train --depth=20 --optimizer-type=soap \
    --soap-lr=<calibrated> --precondition-frequency=<calibrated> \
    --run="cmp_soap" --model-tag="cmp_soap"
```

**关键输出**：三组 wandb 曲线（val_bpb vs step/time/flops）+ CORE metric + 显存峰值。

### 6.3 阶段三：消融实验（~4-6 hr）

针对关键机制设计消融实验：

#### 消融 A：Muon 正交化频率

在 d20 上测试不同正交化频率对收敛的影响：

| 配置 | 正交化频率 | 预期效果 |
|------|-----------|---------|
| 每步正交化（默认） | 1 | 基线 |
| 每 2 步正交化 | 2 | 计算开销减半，质量影响？ |
| 每 5 步正交化 | 5 | 大幅降低开销 |
| 仅衰减期正交化 | warmdown 阶段才启用 | 测试"强心剂假说" |

#### 消融 B：SOAP 预条件频率

| 配置 | precondition_frequency | 预期效果 |
|------|----------------------|---------|
| 频繁更新 | 10 | 更精确的曲率估计，更高计算成本 |
| 中等频率 | 50 | 平衡点 |
| 稀疏更新 | 100 | 最低计算成本，质量影响？ |

#### 消融 C：WSD 衰减比例

| 配置 | warmdown_ratio | 预期效果 |
|------|---------------|---------|
| 短衰减 | 0.25 | 更长稳定探索期 |
| 中等衰减（默认） | 0.5 | 基线 |
| 长衰减 | 0.75 | 更充分的收敛沉淀 |

---

## 7. 显存与算力需求

### 7.1 优化器显存对比（d20，~430M params）

| 优化器 | 参数 | 梯度 | 优化器状态 | 合计（不含激活） |
|--------|------|------|-----------|----------------|
| MuonAdamW | 860 MB | 860 MB | ~870 MB | ~2.6 GB |
| 纯 AdamW | 860 MB | 860 MB | ~1460 MB | ~3.2 GB |
| SOAP | 860 MB | 860 MB | ~1900-2400 MB* | ~3.6-4.1 GB |

*SOAP 额外存储 Kronecker 因子化预条件矩阵（Q_left, Q_right），对 d20 的矩阵形状：
- `(1280, 5120)` → Q_left: 1280×1280 + Q_right: 5120×5120 → ~56 MB/层
- 20 层 × 6 个矩阵 ≈ 额外 ~500-800 MB

### 7.2 GPU 适配

| GPU | VRAM | d12 | d20 | 说明 |
|-----|------|----|-----|------|
| RTX 4090 24GB | 24 GB | ✅ bs≤32 | ⚠️ bs≤4（SOAP 可能 OOM） | d12 推荐 |
| RTX 6000 Ada 48GB | 48 GB | ✅ | ✅ bs≤32 | d20 可运行 |
| **RTX PRO 6000 96GB** | **96 GB** | ✅ | **✅ bs≤64** | **推荐配置，充裕余量** |

### 7.3 总时间预算（单卡 RTX PRO 6000）

| 阶段 | 运行数 | 单次耗时 | 小计 |
|------|-------|---------|------|
| d12 超参搜索 | ~43 | 5-8 min | ~4.5-5.5 hr |
| d12→d20 迁移校验 | 9 | ~5 min | ~0.75 hr |
| d20 正式对比 | 3 | 25-35 min | ~1.5-2 hr |
| d20 消融 | 10 | 25-35 min | ~4-6 hr |
| **总计** | **~65** | | **~11-14 hr** |

---

## 8. 代码实现要点

### 8.1 需修改/新增文件

| 文件 | 变更类型 | 内容 |
|------|---------|------|
| `nanochat/optim.py` | 修改 | 新增 `SOAPAdamW` 类、新增 `PureAdamW` 类 |
| `nanochat/gpt.py` | 修改 | `setup_optimizer()` 增加 `optimizer_type` 参数分发 |
| `scripts/base_train.py` | 修改 | 新增 `--optimizer-type`, `--soap-lr`, `--precondition-frequency` 等 CLI 参数；新增 grad_norm 日志 |
| `runs/optimizer_sweep.sh` | 新增 | 自动化实验编排脚本 |

### 8.2 SOAP 实现要点

```python
class SOAPAdamW:
    """
    SOAP for matrix params, AdamW for non-matrix params.
    
    SOAP 核心流程（每步）:
    1. g = gradient
    2. g_rotated = Q_left.T @ g @ Q_right          # 旋转到特征基
    3. exp_avg = beta1 * exp_avg + (1-beta1) * g_rotated
    4. exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * g_rotated²
    5. update_rotated = exp_avg / sqrt(exp_avg_sq + eps)
    6. update = Q_left @ update_rotated @ Q_right.T  # 旋转回原空间
    7. param -= lr * update
    
    每 precondition_frequency 步额外执行:
    8. 更新 Kronecker 因子: L += g @ g.T, R += g.T @ g
    9. Q_left, Q_right = eigh(L), eigh(R)           # 特征分解
    """
```

### 8.3 SOAP 分块预条件策略

d12 的权重矩阵形状为 `(768, 3072)`，对 3072 维度做完整特征分解的计算量为 O(3072³) ≈ 2.9×10¹⁰ FLOP，不可忽略。实现时需考虑以下策略：

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| **完整特征分解** | 对 m×m 和 n×n 做完整 `torch.linalg.eigh` | 最精确的曲率估计 | 计算量大，P99 延迟高 |
| **分块对角预条件** | 将大矩阵分成 k 个块，各块独立做特征分解 | 计算量降为 O(d³/k²) | 忽略块间相关性 |
| **维度上限截断** | 设置 `max_precond_dim`，超过阈值的维度不做预条件 | 简单直接 | 大权重矩阵退化为普通 Adam |

**本实验采用策略**：完整特征分解 + `precondition_frequency` 拉长（10-50），通过降低更新频率而非降低精度来控制计算成本。在报告中需标注所采用的策略。

### 8.4 训练脚本 CLI 扩展

```bash
# 新增参数
--optimizer-type          # muon | adamw | soap（默认 muon，保持向后兼容）
--soap-lr                 # SOAP 矩阵参数学习率（默认 3e-3）
--precondition-frequency  # SOAP 预条件更新频率（默认 10）
--seed                    # 全局随机种子（默认 42）
```

---

## 9. 结果分析框架

### 9.1 主对比报告模板

```
┌───────────────────────────────────────────────────────────────────┐
│                    优化器对比实验结果（d12）                      │
├──────────────────┬──────────┬──────────┬──────────┬──────────────┤
│ 指标             │ MuonAdamW│ AdamW    │ SOAP     │ 说明         │
├──────────────────┼──────────┼──────────┼──────────┼──────────────┤
│ final_val_bpb    │          │          │          │ ↓ 越低越好   │
│ min_val_bpb      │          │          │          │ ↓ 越低越好   │
│ CORE metric      │          │          │          │ ↑ 越高越好   │
│ gen_gap          │          │          │          │ ↓ 越小越好   │
│ tok/sec          │          │          │          │ ↑ 越高越好   │
│ avg_step_time    │          │          │          │ ↓ 越低越好   │
│ p99_step_time    │          │          │          │ ↓ 越低越好   │
│ step_jitter (CV) │          │          │          │ ↓ 越低越好   │
│ bf16_mfu         │          │          │          │ ↑ 越高越好   │
│ peak_memory      │          │          │          │ ↓ 越低越好   │
│ update/weight    │          │          │          │ 1e-4~1e-2    │
│ loss_spikes      │          │          │          │ ↓ 越少越好   │
│ time_to_tgt      │          │          │          │ ↓ 越短越好   │
│ flops_to_tgt     │          │          │          │ ↓ 越少越好   │
└──────────────────┴──────────┴──────────┴──────────┴──────────────┘
```

### 9.2 关键对比维度

1. **收敛质量 vs 步数**：val_bpb vs step 曲线 → 哪个优化器收敛最快？
2. **收敛质量 vs 挂钟时间**：val_bpb vs wall_time 曲线 → 考虑优化器计算开销后谁更快？
3. **收敛质量 vs FLOPs**：val_bpb vs total_flops 曲线 → 单位计算量的效率谁最高？
4. **效率前沿**：peak_memory vs final_val_bpb 散点图 → 显存-质量帕累托前沿

---

## 10. 预期假设与验证

| 假设编号 | 假设内容 | 验证方法 |
|---------|---------|---------|
| H1 | MuonAdamW 在 step 维度收敛速度优于 AdamW | 对比 steps_to_target |
| H2 | SOAP 在 step 维度收敛速度优于 AdamW（40%+，如论文所述） | 对比 steps_to_target |
| H3 | Muon 的 Polar Express 计算开销使其 tok/sec 低于 AdamW | 对比 tok/sec |
| H4 | SOAP 的预条件矩阵存储导致其 peak_memory 最高 | 对比 peak_memory |
| H5 | 在 wall_time 维度，MuonAdamW 综合效率最优 | 对比 time_to_target |
| H6 | Muon 正交化频率可以降低而不显著损失质量 | 消融 A 的 val_bpb 对比 |
| H7 | WSD 衰减期注入正交化（"强心剂"策略）可以近似全程正交化的收敛质量 | 消融 A 的末尾配置 |
| H8 | SOAP 的步耗时 jitter 显著高于 Muon 和 AdamW（因周期性特征分解） | 对比 p99_step_time 和 step_time_jitter |
| H9 | SOAP/Muon 的 generalization_gap 大于 AdamW（更深的曲率拟合→更易过拟合） | 对比 val_bpb - train_bpb |
| H10 | Muon 各层的 update_to_weight_ratio 分布比 AdamW 更均匀（正交化的等方差效应） | 对比逐层 ‖Δw‖/‖w‖ 的方差 |
