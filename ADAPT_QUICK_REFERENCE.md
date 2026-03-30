# Few-shot Adaptation 快速参考卡

## 命令示例

### 启用 Few-shot 适配 (推荐)
```bash
python run_CNN_EMG.py \
  --dataset mcs \
  --domain_generalization CORAL \
  --leave_one_subject_out True \
  --leftout_subject 1 \
  --adapt_ratio 0.1 \
  --lambda_coral 0.01 \
  --lambda_prototype 1.0 \
  --epochs 25
```

### 禁用 Few-shot 适配 (对照)
```bash
python run_CNN_EMG.py \
  --dataset mcs \
  --domain_generalization CORAL \
  --leave_one_subject_out True \
  --leftout_subject 1 \
  --adapt_ratio 0.0 \
  --lambda_coral 0.01 \
  --lambda_prototype 1.0 \
  --epochs 25
```

### 对所有被试运行 (循环)
```bash
for subject in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== Subject $subject ==="
  python run_CNN_EMG.py \
    --dataset mcs \
    --domain_generalization CORAL \
    --leave_one_subject_out True \
    --leftout_subject $subject \
    --adapt_ratio 0.1
done
```

---

## 参数快速查询

| 参数 | 类型 | 范围 | 默认值 | 说明 |
|------|------|------|--------|------|
| `adapt_ratio` | float | 0.0 ~ 1.0 | 0.0 | 采样比例 |
| `lambda_coral` | float | 0.0 ~ 1.0 | 0.1 | CORAL 损失权重 |
| `lambda_prototype` | float | 0.0 ~ 1.0 | 0.5 | 原型损失权重 |
| `epochs` | int | 1 ~ 100 | 25 | 训练轮数 |
| `batch_size` | int | 16 ~ 256 | 64 | 批大小 |
| `learning_rate` | float | 1e-5 ~ 1e-2 | 1e-4 | 学习率 |
| `seed` | int | ≥ 0 | 0 | 随机种子 |

---

## 手势识别数据集 (EMG) 特定参数

### MCS EMG 数据集
```bash
--dataset mcs
--full_dataset_mcs True    # 使用完整数据集 (默认 False)
--exercises 1,2,3          # 手势类别 (默认全部)
```

### Ninapro DB2/DB5
```bash
--dataset ninapro-db5
--partial_dataset_ninapro True  # 使用部分数据 (默认 False)
```

### CapgMyo
```bash
--dataset capgmyo
--leave_one_session_out False
```

---

## 输出文件位置

### 标准测试结果
```
result_subject_{id}/
├── train_results.pkl
├── validation_results.pkl
├── test_results.pkl
├── train_confusion_matrix.png
├── validation_confusion_matrix.png
├── test_confusion_matrix.png
└── summary.txt
```

### Few-shot 适配结果 (adapt_ratio > 0.0 时)
```
result_subject_{id}_adapt/
├── adaptation_results.pkl
├── adaptation_summary.txt
└── test_adapted_confusion_matrix.png
```

---

## 结果对比脚本

```python
# 对比无适配 vs 有适配
import pickle
import os

def compare_results(subject_id):
    # 加载无适配结果
    with open(f'result_subject_{subject_id}/test_results.pkl', 'rb') as f:
        baseline = pickle.load(f)
    
    # 加载有适配结果
    with open(f'result_subject_{subject_id}_adapt/adaptation_results.pkl', 'rb') as f:
        adapted = pickle.load(f)
    
    print(f"\n=== Subject {subject_id} ===")
    print(f"Baseline Acc:  {baseline['accuracy']:.4f}")
    print(f"Adapted Acc:   {adapted['accuracy']:.4f}")
    print(f"Improvement:   {(adapted['accuracy']-baseline['accuracy'])*100:+.2f}%")
    
    # 逐类改进
    from sklearn.metrics import precision_recall_fscore_support
    baseline_f1 = precision_recall_fscore_support(baseline['labels'], 
                                                  baseline['predictions'])[2].mean()
    adapted_f1 = precision_recall_fscore_support(adapted['labels'], 
                                                 adapted['predictions'])[2].mean()
    print(f"Baseline F1:   {baseline_f1:.4f}")
    print(f"Adapted F1:    {adapted_f1:.4f}")

# 对所有被试进行对比
for subj in range(1, 11):
    try:
        compare_results(subj)
    except FileNotFoundError:
        print(f"Subject {subj}: 结果文件未找到")
```

---

## 常见问题解决

### 1. 支持集太小
```bash
# 增加采样率
--adapt_ratio 0.2  # 从 0.1 改为 0.2
```

### 2. 收敛较慢
```bash
# 增加 CORAL 损失权重
--lambda_coral 0.1  # 从 0.01 改为 0.1
```

### 3. 过拟合
```bash
# 减少 CORAL 和原型损失权重
--lambda_coral 0.001
--lambda_prototype 0.1
```

### 4. 显存不足
```bash
--batch_size 32     # 从 64 改为 32
--adapt_ratio 0.05  # 从 0.1 改为 0.05
```

---

## 期望性能指标

| 适配比例 | 期望精度提升 | 置信度 | 备注 |
|--------|-----------|--------|------|
| 0.0% | - | - | 基准(无适配) |
| 5% | +1-3% | 中等 | 适用大数据集 |
| 10% | +2-5% | 高 | 推荐配置 |
| 20% | +3-6% | 高 | 适用小数据集 |
| > 30% | +3-5% | 中等 | 收益饱和 |

**注**：实际改进取决于跨被试域间隙大小，EMG 通常为 5-15%

---

## 可视化分析

### 对比混淆矩阵
```python
import matplotlib.pyplot as plt
import pickle

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 无适配
with open('result_subject_1/test_results.pkl', 'rb') as f:
    baseline = pickle.load(f)
    axes[0].imshow(baseline['confusion_matrix'], cmap='Blues')
    axes[0].set_title('Baseline (No Adaptation)')

# 有适配
with open('result_subject_1_adapt/adaptation_results.pkl', 'rb') as f:
    adapted = pickle.load(f)
    axes[1].imshow(adapted['confusion_matrix'], cmap='Greens')
    axes[1].set_title(f'Adapted (adapt_ratio={adapted["adapt_ratio"]})')

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
```

---

## 配置文件示例 (config/example_adapt.yaml)

```yaml
# Few-shot Adaptation Configuration
dataset: mcs
domain_generalization: CORAL
leave_one_subject_out: true
leftout_subject: 1

# Few-shot 参数
adapt_ratio: 0.1

# CORAL 和原型损失权重
lambda_coral: 0.01
lambda_prototype: 1.0

# 训练参数
epochs: 25
batch_size: 64
learning_rate: 0.0001
seed: 0

# 数据集相关
full_dataset_mcs: false
exercises: [1, 2, 3]

# GPU 和多进程
gpu: 0
multiprocessing: true
```

使用方法：
```bash
python run_CNN_EMG.py --config config/example_adapt.yaml
```

---

## 关键检查清单

- [ ] 参数中 `adapt_ratio > 0.0` (启用适配)
- [ ] `--leave_one_subject_out True` (LOSO 模式)
- [ ] `--domain_generalization CORAL` (使用 CORAL)
- [ ] 数据集存在且路径正确
- [ ] 验证/测试集规模合理 (> adapt_ratio * 100 个样本)
- [ ] GPU 显存充足 (建议 ≥ 6GB)
- [ ] 确认输出目录可写入

---

更多信息见 `FEWSHOT_ADAPTATION_GUIDE.md`
