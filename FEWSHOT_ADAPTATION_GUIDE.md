# Few-shot + Transductive Domain Adaptation 使用指南

## 概述

这是一个基于 LOSO CORAL 框架的"可控比例 few-shot + transductive 适配"测试流程。通过从验证集和测试集中采样少量样本，估计未训练的Subject 0（测试被试）的特征分布，然后在测试时对特征进行对齐，从而改进跨被试的准确率。

---

## 核心原理

### 问题设置
- **LOSO 设置**：Subject 1-9 用于训练，Subject 0 进行保留测试
- **未训练被试**：Subject 0 在训练阶段完全未见，统计分布未知
- **目标**：利用少量 Subject 0 数据估计其分布，对齐到已知的训练空间

### 适配流程

```
1. 训练阶段 (不变)
   └─ Subject 1-9 训练 → 学习分布特性

2. 支持集构建
   ├─ Validation 采样: val_support_size = len(val) × adapt_ratio
   ├─ Test 采样: test_support_size = len(test) × adapt_ratio
   └─ 取最小值: support_size = min(val_support_size, test_support_size)

3. Subject 0 分布估计
   ├─ 从支持集提取特征 [support_size×2, 512]
   ├─ 计算 mean_0 (Subject 0 均值) [512]
   └─ 计算 var_0 (Subject 0 方差) [512]

4. 特征对齐与测试
   ├─ 移除 test_support 样本
   ├─ 对剩余测试样本进行特征对齐
   └─ 在对齐后特征空间进行分类
```

### 特征对齐公式

对于任意测试样本的特征 $\mathbf{f}$：

$$\mathbf{f}_{aligned} = \frac{\mathbf{f} - \boldsymbol{\mu}_0}{\sqrt{\mathbf{v}_0 + \epsilon}} \cdot \sqrt{\mathbf{v}_{train} + \epsilon} + \boldsymbol{\mu}_{train}$$

其中：
- $\boldsymbol{\mu}_0, \mathbf{v}_0$：Subject 0 的均值和对角方差（从支持集估计）
- $\boldsymbol{\mu}_{train}, \mathbf{v}_{train}$：训练集 (Subject 1-9) 的均值和对角方差
- $\epsilon = 1e-5$：数值稳定性常数

---

## 使用方法

### 1. 基本用法

```bash
python run_CNN_EMG.py \
    --dataset mcs \
    --domain_generalization CORAL \
    --leave_one_subject_out True \
    --leftout_subject 1 \
    --epochs 25 \
    --lambda_coral 0.1 \
    --lambda_prototype 0.5 \
    --adapt_ratio 0.1
```

### 2. 参数说明

**新增参数：**
- `--adapt_ratio` (float, 默认值: 0.0)
  - 范围：0.0 ~ 1.0
  - 含义：从 validation 和 test 中各采样该比例的数据
  - 0.0：禁用 few-shot 适配；> 0.0：启用适配

**相关现有参数：**
- `--lambda_coral`：CORAL 损失权重 (推荐 0.01 ~ 0.1)
- `--lambda_prototype`：原型损失权重 (推荐 0.5 ~ 1.0)
- `--seed`：随机种子，影响支持集采样

### 3. 推荐配置

#### 小规模数据集 (< 2000 样本/被试)
```bash
--adapt_ratio 0.2  # 采样率 20%
--lambda_coral 0.01
--lambda_prototype 1.0
```

#### 中规模数据集 (2000-5000 样本/被试)
```bash
--adapt_ratio 0.1   # 采样率 10%
--lambda_coral 0.05
--lambda_prototype 0.5
```

#### 大规模数据集 (> 5000 样本/被试)
```bash
--adapt_ratio 0.05  # 采样率 5%
--lambda_coral 0.1
--lambda_prototype 0.5
```

---

## 输出结果

### 目录结构

```
result_subject_{id}_adapt/
├── adaptation_results.pkl          # 完整的适配结果 (pickle格式)
├── adaptation_summary.txt          # 文本摘要
└── test_adapted_confusion_matrix.png  # 混淆矩阵可视化
```

### PKL 文件内容

```python
{
    'adapt_ratio': 0.1,                    # 适配比例
    'support_size': 50,                    # 支持集大小
    'val_support_indices': [...],          # validation 支持集索引
    'test_support_indices': [...],         # test 支持集索引
    'test_remaining_indices': [...],       # 用于测试的剩余索引
    
    'mean_train': [...],                   # 训练集均值 [512]
    'var_train': [...],                    # 训练集方差 [512]
    'mean_0': [...],                       # Subject 0 均值 [512]
    'var_0': [...],                        # Subject 0 方差 [512]
    
    'predictions': [...],                  # 预测标签
    'labels': [...],                       # 真实标签
    'features_aligned': [...],             # 对齐后的特征
    'accuracy': 0.85,                      # 准确率
    'confusion_matrix': [...],             # 混淆矩阵
    'gesture_labels': [...]                # 手势标签列表
}
```

### 关键约束验证

运行日志会输出以下验证信息：

```
[Step 2] Building support set with adapt_ratio=0.1...
  Val set size: 512, support_size: 51
  Test set size: 512, support_size: 51
  Final support_size (min): 51          # ✓ 严格相等
  Sampled 51 from validation
  Sampled 51 from test

[Step 5] Testing with feature alignment...
  Original test set size: 512
  Test support removed: 51              # ✓ 从测试中移除
  Remaining test set size: 461          # ✓ 不重复使用
```

---

## 实现细节

### 零梯度约束

```python
with torch.no_grad():
    _, feat = self.model(X_batch)  # 仅前向，无反向传播
    feat_aligned = align_features(feat)
    logits = classifier(feat_aligned)
    # ✓ 不编辑模型权重
```

### 对角方差实现

```python
mean_0 = support_features.mean(axis=0)        # [512]
var_0 = np.var(support_features, axis=0)     # [512] 对角线
# ✓ 使用对角线元素，避免完整协方差矩阵计算
```

### 样本移除验证

```python
val_support_indices = np.random.choice(val_indices_all, size=support_size)
test_support_indices = np.random.choice(test_indices_all, size=support_size)

test_indices_remain = np.setdiff1d(test_indices_all, test_support_indices)
# ✓ 确保 test_support 中的样本不在 test_indices_remain 中
```

---

## 对比分析

### 无适配 vs 有适配

| 指标 | 无适配 (adapt_ratio=0.0) | 有适配 (adapt_ratio=0.1) |
|-----|-------------------------|------------------------|
| 支持集大小 | 0 | support_size × 2 |
| 分布估计 | 基于训练集 (Subject 1-9) | 包含 Subject 0 数据 |
| 特征对齐 | 无 | 是 (按对角方差) |
| 样本复用 | 测试集全量使用 | 支持集移除 |

### 期望改进

在 EMG 跨被试场景中，few-shot 适配通常可以改进：
- **小样本适配**：3-5 个手势周期 (100-200ms) 可显著提升精度
- **减小域间隙**：Subject 0 的分布偏差通常造成 5-15% 的准精度下降
- **渐进式收益**：adapt_ratio 从 0.05 → 0.1 → 0.2 时收益递减

---

## 故障排查

### Q: 报错 "Index out of bounds"
**A:** 检查 `adapt_ratio` 不超过 1.0；validation/test 数据不为空

### Q: 支持集大小为 1
**A:** 数据集过小；调整 `adapt_ratio` 使支持集 ≥ 10 个样本推荐

### Q: 准确率反而下降
**A:** 
- 增大 `adapt_ratio` (可能支持集太小)
- 检查 `lambda_coral` 和 `lambda_prototype` 配置
- 验证训练集收敛状态

### Q: 内存溢出
**A:** 
- 减小 batch_size (默认 64)
- 减小 `adapt_ratio`

---

## 参考论文与方法

此实现基于以下思想的融合：

1. **Deep CORAL** (Sun et al., 2016)
   - 通过对齐分布均值和协方差进行域适配

2. **Transductive Transfer Learning**
   - 利用未标记的目标域数据估计分布

3. **Few-shot Domain Adaptation**
   - 用少量标记样本快速适配

---

## 代码示例：加载和分析结果

```python
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载适配结果
with open('result_subject_1_adapt/adaptation_results.pkl', 'rb') as f:
    results = pickle.load(f)

# 基本指标
print(f"适配比例: {results['adapt_ratio']}")
print(f"支持集大小: {results['support_size']}")
print(f"测试准确率: {results['accuracy']:.4f}")

# 特征对齐效果
aligned_features = results['features_aligned']
print(f"对齐特征形状: {aligned_features.shape}")
print(f"对齐特征均值范数: {np.linalg.norm(aligned_features.mean(axis=0)):.4f}")

# 逐类性能
from sklearn.metrics import classification_report
print(classification_report(results['labels'], results['predictions'], 
                          target_names=results['gesture_labels']))
```

---

## 版本历史

- **v1.0** (2026-03-30)：初始实现，支持对角方差对齐、支持集移除、严格相等采样

---

## 联系与反馈

如有问题或建议，请参考主项目说明文档。
