# Few-shot Adaptation 高级用法与优化指南

## 1. 深层原理解析

### 1.1 特征空间对齐的数学基础

对于任意样本 $\mathbf{x}$ 的特征 $\mathbf{f} = \Phi(\mathbf{x}) \in \mathbb{R}^{512}$，目标是将其分布从 Subject 0 空间对齐到训练空间。

**标准化步骤：**

$$\text{step1}: \mathbf{f}_{centered} = \mathbf{f} - \boldsymbol{\mu}_0$$

$$\text{step2}: \mathbf{f}_{normalized} = \frac{\mathbf{f}_{centered}}{\sqrt{\mathbf{v}_0 + \epsilon}}$$

$$\text{step3}: \mathbf{f}_{aligned} = \mathbf{f}_{normalized} \cdot \sqrt{\mathbf{v}_{train} + \epsilon} + \boldsymbol{\mu}_{train}$$

**解释：**
- Step 1：移除 Subject 0 的偏置
- Step 2：除以 Subject 0 的标准差（使方差为 1）
- Step 3：乘以训练集的标准差并加上训练集均值（匹配尺度和均值）

### 1.2 为什么使用对角方差

**完整协方差的问题：**
- 维度灾难：$512 \times 512 = 262144$ 元素
- 支持集估计不稳定（特别是 < 100 样本时）
- 计算和存储成本高

**对角方差的优势：**
- 仅 512 个参数
- 假设维度独立（在实践中可行）
- 稀疏支持集上也能稳定估计
- 快速计算

**理论保证：**
当 EMG 特征通过 ResNet18 提取时，特征空间中：
- 不同维度的方差差异大（来自不同卷积核）
- 跨维度的协方差相对较小（batch norm 已去相关）

---

## 2. 超参数调优

### 2.1 adapt_ratio 的影响

| adapt_ratio | 支持集大小 | 估计方差 | 时间开销 | 推荐场景 |
|------------|----------|--------|--------|---------|
| 0.01 | ~5 | 高 | 低 | 超大数据集 |
| 0.05 | ~25 | 中高 | 低 | 大数据集 |
| 0.10 | ~50 | 中 | 中 | **推荐** |
| 0.20 | ~100 | 低 | 中高 | 小数据集 |
| 0.30+ | ~150+ | 很低 | 高 | 超小数据集 |

**经验法则：** 支持集大小 = max(20, 样本数 × adapt_ratio)

### 2.2 lambda_coral vs lambda_prototype

```
总损失 = CE + lambda_coral × CORAL + lambda_prototype × Prototype
```

| 配置 | lambda_coral | lambda_prototype | 适用 | 特点 |
|------|-------------|-----------------|------|------|
| 保守 | 0.001 | 0.1 | 稳定训练 | 改进小 |
| 平衡 | 0.01 | 0.5 | 通用 | **推荐** |
| 激进 | 0.1 | 1.0 | 强约束 | 可能过度正则 |

**调优策略：**

1. **跨被试间隙大（> 10% 精度下降）**
   ```
   lambda_coral: 0.05-0.1
   lambda_prototype: 0.5-1.0
   ```

2. **跨被试间隙中等（5-10%）**
   ```
   lambda_coral: 0.01-0.05
   lambda_prototype: 0.5
   ```

3. **跨被试间隙小（< 5%）**
   ```
   lambda_coral: 0.001-0.01
   lambda_prototype: 0.1-0.5
   ```

### 2.3 学习率与 Batch 大小

**LOSO 特定配置：**

```python
# 每个 Subject 的批次数较少时
batch_size = 32  # 较小的批次
learning_rate = 5e-4  # 较大的学习率

# 数据充足时
batch_size = 64  # 标准批次
learning_rate = 1e-4  # 标准学习率
```

---

## 3. 诊断和调试

### 3.1 支持集质量评估

```python
import numpy as np
import pickle

# 加载适配结果
with open('result_subject_1_adapt/adaptation_results.pkl', 'rb') as f:
    results = pickle.load(f)

# 1. 检查均值偏差
mean_diff = np.linalg.norm(results['mean_0'] - results['mean_train'])
print(f"均值偏差: {mean_diff:.4f}")
# 期望: < 5.0 (好), < 10.0 (可接受), > 15.0 (可能问题)

# 2. 检查方差比例
var_ratio = results['var_0'] / (results['var_train'] + 1e-5)
print(f"方差比例: {var_ratio.mean():.4f}, 范围: [{var_ratio.min():.4f}, {var_ratio.max():.4f}]")
# 期望: 接近 1.0 (好), 0.5-2.0 (可接受), > 3.0 (异常)

# 3. 检查特征对齐效果
aligned_feat = results['features_aligned']
aligned_mean = aligned_feat.mean(axis=0)
aligned_var = np.var(aligned_feat, axis=0)
print(f"对齐后均值范数: {np.linalg.norm(aligned_mean):.4f}")
print(f"对齐后方差范围: [{aligned_var.min():.6f}, {aligned_var.max():.6f}]")
```

### 3.2 训练过程监控

```python
# 添加到 LOSO_CORAL_Trainer 中进行实时监控
def monitor_adaptation(self):
    """在适配期间进行监控"""
    print(f"[Monitor] Training distribution:")
    print(f"  Mean norm: {np.linalg.norm(self.subject_stats_memory.mean[0]):.4f}")
    print(f"  Var range: [{self.subject_stats_memory.cov[0].diag().min():.6f}, "
          f"{self.subject_stats_memory.cov[0].diag().max():.6f}]")
    
    # 检查特定手势的准确率
    from sklearn.metrics import recall_score
    recalls = recall_score(self.test_true_labels, self.test_predictions, 
                          average=None)
    print(f"  Per-class recall: {recalls}")
```

### 3.3 常见问题诊断

**问题 1：支持集均值偏离训练均值很远**
```
原因: 支持集样本偏离 Subject 0 真实分布
解决: 
  - 增加 adapt_ratio (采样更多样本)
  - 检查数据标签是否错误
  - 检查数据预处理是否一致
```

**问题 2：对齐后准确率反而下降**
```
原因: 分布差异太大，简单线性对齐不足
解决:
  - 增加 adapt_ratio 改进分布估计
  - 调整 lambda_coral 和 lambda_prototype
  - 考虑增加训练轮数
```

**问题 3：支持集大小不等**
```
原因: adapt_ratio × len(val) 和 adapt_ratio × len(test) 向下取整后不同
解决: 自动处理 (取最小值)，可在日志中验证
```

---

## 4. 性能优化

### 4.1 计算效率优化

```python
# 优化 1: 使用 torch.cuda.amp 混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output, features = self.model(X_batch)
    loss = compute_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)

# 优化 2: 特征提取缓存 (对于 few-shot)
@torch.no_grad()
def extract_features_batch(self, data_loader):
    """一次性提取所有特征并缓存"""
    features_all = []
    for X_batch, _ in data_loader:
        X_batch = X_batch.to(self.device).to(torch.float32)
        _, feat = self.model(X_batch)
        features_all.append(feat.cpu().numpy())
    return np.concatenate(features_all, axis=0)

# 优化 3: 分布统计向量化计算
def compute_stats_vectorized(features):
    """使用 numpy 向量化而非循环"""
    mean = features.mean(axis=0)  # [512]
    var = ((features - mean)**2).mean(axis=0)  # [512]
    return mean, var
```

### 4.2 内存优化

```python
# 当 adapt_ratio 很大时的内存优化
def process_support_set_streaming(self, support_indices, batch_size=32):
    """流式处理支持集，避免一次性加载全部"""
    features_all = []
    
    for i in range(0, len(support_indices), batch_size):
        batch_indices = support_indices[i:i+batch_size]
        features_batch = self._extract_features(batch_indices)
        features_all.append(features_batch)
        
        # 及时释放内存
        torch.cuda.empty_cache()
    
    return np.concatenate(features_all, axis=0)
```

### 4.3 并行化

```bash
# 对多个被试并行运行测试
parallel -j 4 'python run_CNN_EMG.py \
    --leftout_subject {} \
    --adapt_ratio 0.1' ::: 1 2 3 4 5 6 7 8 9 10

# 使用 GNU parallel (需要先安装)
# 或者使用 xargs (简化版)
seq 1 10 | xargs -P 4 -I {} bash -c \
    'python run_CNN_EMG.py --leftout_subject {} --adapt_ratio 0.1'
```

---

## 5. 高级配置

### 5.1 自适应 adapt_ratio

```python
def adaptive_ratio_selector(dataset_size, num_gestures):
    """根据数据规模自动选择 adapt_ratio"""
    samples_per_gesture = dataset_size / num_gestures
    
    if samples_per_gesture < 200:
        return 0.3
    elif samples_per_gesture < 500:
        return 0.2
    elif samples_per_gesture < 1000:
        return 0.1
    else:
        return 0.05
```

### 5.2 多阶段适配

```python
"""
第一阶段: 基准训练 (adapt_ratio=0.0)
第二阶段: Few-shot 适配 (adapt_ratio=0.1)
第三阶段: 可选的在线适配 (adapt_ratio=1.0)
"""

# 配置文件
stages = [
    {'name': 'baseline', 'adapt_ratio': 0.0, 'epochs': 25},
    {'name': 'fewshot', 'adapt_ratio': 0.1, 'epochs': 10},
    {'name': 'online', 'adapt_ratio': 1.0, 'epochs': 5},
]
```

### 5.3 交叉验证支持集

```python
from sklearn.model_selection import KFold

def cross_validate_adaptation(val_dataset, test_dataset, adapt_ratio=0.1, k=5):
    """使用 K-fold 交叉验证支持集"""
    
    num_val = len(val_dataset)
    support_size = int(num_val * adapt_ratio)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    accuracies = []
    for fold, (_, test_idx) in enumerate(kf.split(np.arange(num_val))):
        support_indices = test_idx[:support_size]
        val_support = val_dataset[support_indices]
        
        # 运行适配并评估
        acc = run_adaptation_fold(val_support, test_dataset)
        accuracies.append(acc)
    
    return {
        'mean_acc': np.mean(accuracies),
        'std_acc': np.std(accuracies),
        'accuracies': accuracies,
    }
```

---

## 6. 实验最佳实践

### 6.1 标准实验流程

```bash
#!/bin/bash
# 标准化实验脚本

DATASET="mcs"
ADAPT_RATIOS="0.0 0.05 0.1 0.15 0.2"
SUBJECTS=$(seq 1 10)

# 日志目录
LOG_DIR="experiments/few_shot_exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# 逐被试逐适配比例运行
for subject in $SUBJECTS; do
    for ratio in $ADAPT_RATIOS; do
        echo "Running subject=$subject, adapt_ratio=$ratio"
        
        python run_CNN_EMG.py \
            --dataset $DATASET \
            --domain_generalization CORAL \
            --leave_one_subject_out True \
            --leftout_subject $subject \
            --adapt_ratio $ratio \
            --lambda_coral 0.01 \
            --lambda_prototype 1.0 \
            --epochs 25 \
            --seed 42 \
            2>&1 | tee "$LOG_DIR/subject_${subject}_ratio_${ratio}.log"
    done
done

echo "All experiments completed. Logs in: $LOG_DIR"
```

### 6.2 结果统计脚本

```python
import os
import pickle
import numpy as np
import pandas as pd

def summarize_experiments(log_dir):
    """汇总所有实验结果"""
    
    results = []
    
    for subject in range(1, 11):
        for ratio in [0.0, 0.05, 0.1, 0.15, 0.2]:
            result_dir = f'result_subject_{subject}'
            
            if ratio == 0.0:
                pkl_file = f'{result_dir}/test_results.pkl'
            else:
                result_dir += '_adapt'
                pkl_file = f'{result_dir}/adaptation_results.pkl'
            
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    res = pickle.load(f)
                
                results.append({
                    'subject': subject,
                    'adapt_ratio': ratio,
                    'accuracy': res['accuracy'],
                    'support_size': res.get('support_size', 0),
                })
    
    df = pd.DataFrame(results)
    
    # 生成透视表
    pivot = df.pivot_table(values='accuracy', index='subject', 
                            columns='adapt_ratio', aggfunc='mean')
    print("\n准确率概览:")
    print(pivot)
    
    # 计算改进
    improvement = (pivot[0.1] - pivot[0.0]).mean()
    print(f"\n平均改进 (adapt_ratio=0.1): {improvement*100:+.2f}%")
    
    return df

# 使用
df_results = summarize_experiments('.')
df_results.to_csv('adaptation_results_summary.csv', index=False)
```

---

## 7. 常见错误和解决方案

| 错误信息 | 原因 | 解决方案 |
|--------|------|--------|
| `RuntimeError: CUDA out of memory` | 显存不足 | 减少 batch_size 或 adapt_ratio |
| `IndexError: index 512 is out of bounds` | 特征维度错误 | 检查模型输出特征维度 |
| `FileNotFoundError: test_results.pkl` | 基准测试未完成 | 先运行 adapt_ratio=0.0 |
| `ValueError: cannot reshape` | 数据形状不匹配 | 检查 transform 是否正确应用 |
| `AssertionError: support_size < 1` | 数据集太小 | 增加 adapt_ratio 或数据量 |

---

## 8. 发布建议

### 论文中的表述

> We propose a *few-shot transductive domain adaptation* approach for cross-subject EMG recognition. 
> Specifically, we sample $\alpha$ proportion of validation and test data as a support set to estimate the 
> target subject's feature distribution. Features are then aligned via diagonal variance normalization:
> 
> $$\mathbf{f}_{aligned} = \frac{\mathbf{f} - \boldsymbol{\mu}_0}{\sqrt{\mathbf{v}_0 + \epsilon}} \cdot \sqrt{\mathbf{v}_{train} + \epsilon} + \boldsymbol{\mu}_{train}$$
>
> This approach requires no additional backpropagation and achieves {X}% relative improvement.

### 超参数报告

```
Few-shot Adaptation Configuration:
- Adaptation ratio: 0.1
- Support set size: 50 (validation) + 50 (test) = 100
- Feature alignment method: diagonal variance normalization
- Hyperparameters: λ_coral = 0.01, λ_prototype = 1.0
- No additional training required
```

---

更多信息见 `FEWSHOT_ADAPTATION_GUIDE.md` 和 `ADAPT_QUICK_REFERENCE.md`
