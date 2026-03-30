# LOSO CORAL Trainer Implementation

## Overview

The `LOSO_CORAL_Trainer` implements a novel training approach for cross-subject EMG classification in the Leave-One-Subject-Out (LOSO) setting. It combines:

1. **Subject-level Statistics Memory** - Maintains running statistics (mean and covariance) for each subject
2. **CORAL Loss** - Computes distribution alignment loss using stored subject statistics
3. **Prototype Loss** - Applies metric learning to align class semantics across subjects
4. **Subject-Constrained Batching** - Ensures each batch contains samples from only one subject

## Key Components

### 1. SubjectStatisticsMemory

Maintains running statistics for each subject during training:

```
Attributes:
  - mean: [num_subjects, 512] - Running feature mean per subject
  - cov: [num_subjects, 512, 512] - Running covariance matrix per subject
  - updated: [num_subjects] - Track which subjects have been updated
  - ema_alpha: 0.9 - EMA weight for old statistics

Methods:
  - update(subject_id, batch_features): Update subject's statistics using EMA
  - get_target_distribution(subject_id, exclude_current=True): 
      Returns averaged statistics of other subjects (for CORAL loss)
```

**Update Rule (EMA)**:
```
new_mean = ema_alpha * old_mean + (1 - ema_alpha) * batch_mean
new_cov = ema_alpha * old_cov + (1 - ema_alpha) * batch_cov
```

### 2. CORAL Loss (CorralLoss)

Computes Deep CORAL loss between source and target distributions:

```
loss = mean_diff + cov_diff
  where:
    mean_diff = mean((source_mean - target_mean)^2)
    cov_diff = mean((source_cov - target_cov)^2)  [Frobenius norm]
```

**Flow in Training**:
1. Get current batch features from subject A
2. Compute batch mean and covariance
3. Retrieve target statistics: average of all OTHER subjects
4. Compute CORAL loss between batch and target
5. Update subject A's statistics in memory

### 3. Prototype Loss (PrototypeLoss)

Metric learning loss using class prototypes:

```
Steps:
1. Compute class prototypes: mean feature per class in batch
2. Normalize features and prototypes for cosine similarity
3. Compute similarity scores: [batch_size, num_classes]
4. Apply cross-entropy loss on similarity scores
```

This acts as auxiliary loss to align class semantics across subjects.

### 4. Batch Composition

`SingleSubjectPerBatchSampler` ensures each batch contains only one subject:

- Iterates through subjects sequentially
- Creates full batches from each subject's data
- Shuffles within subject to improve generalization
- Skips partial batches (keeps only full batch_size)

**Example** (10 subjects, 64 batch size):
```
Subject 0: batches 1-20 (1280 samples / 64)
Subject 1: batches 21-40
...
Subject 9: batches 181-200
```

### 5. Subject ID Tracking

`CustomDatasetWithSubjectID` extends the base Dataset to track subject membership:

```
Returns: (X, Y, subject_id) for each sample
  - Computed from cumulative_sizes
  - Allows proper CORAL loss computation in training loop
```

## Training Loop

```python
For each epoch:
  For each batch from train_loader:
    1. Unpack batch: X, Y, subject_id (current_subject_id)
    
    2. Forward pass: output, features = model(X)
    
    3. Compute classification loss:
       loss_ce = CrossEntropyLoss(output, labels)
    
    4. Compute batch statistics:
       batch_mean = features.mean()
       batch_cov = features.cov()
    
    5. Get target distribution (exclude current subject):
       target_mean, target_cov = memory.get_target_distribution(
           current_subject_id, exclude_current=True
       )
    
    6. Compute CORAL loss:
       loss_coral = coral_loss(batch_mean, batch_cov, 
                               target_mean, target_cov)
    
    7. Compute prototype loss:
       loss_prototype = prototype_loss(features, labels)
    
    8. Total loss:
       loss = loss_ce + 0.1*loss_coral + 0.5*loss_prototype
    
    9. Backward pass and update weights
    
    10. Update subject memory:
        memory.update(current_subject_id, features)
```

## Loss Weights

Hyperparameters (configurable in code):

```python
lambda1 = 0.1    # CORAL loss weight
lambda2 = 0.5    # Prototype loss weight
ema_alpha = 0.9  # EMA weight for statistics updates
```

These can be adjusted based on your validation performance:
- **Increase lambda1**: More emphasis on cross-subject distribution alignment
- **Increase lambda2**: More emphasis on class semantic alignment
- **Increase ema_alpha**: Longer memory of past subject statistics

## Usage

### Configuration

Create a YAML config (or use existing with modifications):

```yaml
dataset: capgmyo
model: resnet18
domain_generalization: CORAL
leave_one_subject_out: True
batch_size: 64
epochs: 100
learning_rate: 0.0001
```

### Running Training

```bash
# Using config file
python run_CNN_EMG.py --config config/loso_coral_capgmyo.yaml

# Or using command line
python CNN_EMG.py --dataset capgmyo --model resnet18 \
                  --domain_generalization CORAL \
                  --leave_one_subject_out True \
                  --batch_size 64 --epochs 100
```

### Expected Output

Training should show:
- **Total Loss**: Decreases with training
- **CE Loss**: Classification accuracy improving
- **CORAL Loss**: Decreasing as distributions align
- **Proto Loss**: Providing additional regularization
- **Batch Acc**: Subject-wise accuracy improving

Example output:
```
Epoch 50/100
Batch Loss: Total=2.34 CE=1.15 CORAL=0.08 Proto=0.65, Batch Acc: 0.78
```

## Data Requirements

For proper functioning:

1. **LOSO Split**: Ensure training data contains 9 subjects (10 - 1 left-out)
2. **Balanced Data**: Each subject should have sufficient samples (1280+ recommended)
3. **Batch Size**: Should be smaller than subject's sample count (e.g., 64 < 1280)
4. **Cumulative Sizes**: Must be properly set in X_Data to track subject boundaries

## Mathematical Details

### CORAL Loss Computation

Given source distribution (current batch) and target distribution (other subjects):

```
D_A = database of source domain A
D_T = averaged database of target domains

Mean Alignment:
  μ_A = mean(features_A)
  μ_T = mean(features_T)
  
Covariance Alignment:
  Σ_A = (1/n_A) * H_A^T * H_A, where H_A = features_A - μ_A
  Σ_T = (1/n_T) * H_T^T * H_T, where H_T = features_T - μ_T

CORAL Loss:
  L_CORAL = (1/4d²) * ||Σ_A - Σ_T||_F²
  
  (In our implementation: simplified to MSE for efficiency)
```

### Prototype Loss

```
For each class c:
  p_c = mean(features where label==c)  # Class prototype

For each sample i with label y_i:
  similarity_i = cosine_similarity(feature_i, all_prototypes)
  sim_i_j = (feature_i · p_j) / (||feature_i|| * ||p_j||)

Loss:
  L_PROTO = CrossEntropyLoss(similarity, labels)
```

## Advantages

1. **Cross-Subject Alignment**: CORAL loss explicitly aligns feature distributions
2. **Class Semantics**: Prototype loss ensures class prototypes are consistent
3. **Per-Subject Tracking**: Maintains separate statistics for each subject
4. **Stable Training**: EMA updates provide stable statistics accumulation
5. **LOSO Compliant**: Single-subject-per-batch ensures no test leakage

## Known Limitations

1. **First Batch**: Early batches may have high CORAL loss (no good target yet)
2. **Subject Order**: Sampler processes subjects sequentially, may affect convergence
3. **Covariance Computation**: Full covariance can be unstable; diagonal version available
4. **Memory Usage**: Storing covariance matrices adds overhead

## Future Improvements

1. **Diagonal Covariance**: Replace full covariance with diagonal for efficiency
2. **Adaptive Weights**: Adjust lambda weights based on training phase
3. **Subject Scheduling**: Intelligent ordering of subjects for training
4. **Batch Refinement**: Mix subjects in later epochs for final fine-tuning
5. **Validation Metrics**: Add CORAL-specific metrics (domain distance, etc.)

## Troubleshooting

### Issue: CORAL loss stays high

**Solution**: 
- Check if subject statistics are being updated (verify `memory.updated`)
- Increase target subjects to improve target distribution estimation
- Increase lambda1 to emphasize CORAL loss

### Issue: Prototype loss becomes NaN

**Solution**:
- Ensure all classes appear in at least 1 batch
- Reduce learning rate slightly
- Add small epsilon to prototype normalization

### Issue: Poor generalization on left-out subject

**Solution**:
- Increase lambda1 (more CORAL emphasis)
- Increase epochs for better statistics accumulation
- Check batch size appropriateness (should be < subject's data size)

## References

1. Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016)
   - Sun et al. https://arxiv.org/abs/1607.01719

2. Leave-One-Subject-Out Cross-Validation for EMG Classification
   - Common evaluation protocol in EMG/BCI research

## Author

LOSO_CORAL_Trainer implementation for EMG benchmarking project.
