# LOSO CORAL Technical Specification

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Training Loop                          │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   [Backbone]    [Feature Extract]  [Classifier]
   ResNet18         (512-dim)        (→ logits)
        │              │              │
        └──────────────┼──────────────┘
                       │
        ┌──────────────┼────────────────────────┐
        │              │                        │
        ▼              ▼                        ▼
   [ClassLoss]   [CoralLoss]            [PrototypeLoss]
   CrossEntropy   (Frobenius Norm)      (Cosine Sim + CE)
        │              │                        │
        │              ▼                        │
        │         [SubjectMemory]               │
        │         ├─ Mean (EMA)                 │
        │         ├─ Covariance (EMA)          │
        │         └─ Per-subject tracking       │
        │              │                        │
        └──────────────┼────────────────────────┘
                       │
                    [Loss Combination]
                loss = CE + 0.1*CORAL + 0.5*Proto
                       │
                       ▼
                  [Backward Pass]
                  [Update Weights]
```

## Detailed Algorithm

### Phase 1: Batch Preparation (via SingleSubjectPerBatchSampler)

```
Input: Dataset with N subjects, each with S samples
Output: Ordered sequence of batches (B samples each)

Algorithm:
  1. Create subject_indices[i] = list of sample indices for subject i
  2. For each subject i in [0, N-1]:
       a. Shuffle subject_indices[i]
       b. For each batch_start in range(0, len(indices), B):
            - Append batch = indices[batch_start:batch_start+B]
            - (skip if len(batch) < B)
  3. Return flattened list of all batch indices
```

Time Complexity: O(N*S*log(S)) for shuffling
Space Complexity: O(N*S) to store indices

### Phase 2: Forward Pass

```
Input: Batch (B samples from subject k)
Output: Predictions, Features (B×512)

Forward:
  features = backbone(X)  # [B, 512] from ResNet18
  logits = classifier(features)  # [B, num_classes]
```

### Phase 3: Loss Computation

#### 3.1 Classification Loss (Standard)

```
loss_ce = CrossEntropyLoss(logits, labels)
          = mean(-log(softmax(logits)[y_i]))
```

#### 3.2 CORAL Loss (Distribution Alignment)

```
Algorithm:
  1. Compute batch statistics:
     - batch_mean = mean(features, dim=0)  # [512]
     - H = features - batch_mean  # Center: [B, 512]
     - batch_cov = H.T @ H / (B-1)  # [512, 512]
  
  2. Retrieve target statistics from SubjectStatisticsMemory:
     - target_mean, target_cov = memory.get_target_distribution(
         current_subject_id, exclude_current=True
       )
     - Averages mean and cov of all subjects except current
  
  3. Compute CORAL loss:
     - mean_diff = mean((batch_mean - target_mean)²)
     - cov_diff = mean((batch_cov - target_cov)²)
     - loss_coral = mean_diff + cov_diff
  
  4. Update subject statistics using EMA:
     - new_mean = 0.9 * old_mean + 0.1 * batch_mean
     - new_cov = 0.9 * old_cov + 0.1 * batch_cov
```

Mathematical Form:
```
Σ_A (source) = cov(features_A)
Σ_T (target) = mean(cov(features_T) for T in other_subjects)

loss_CORAL = ||Σ_A - Σ_T||_F² + ||μ_A - μ_T||²
           where ||·||_F is Frobenius norm
```

#### 3.3 Prototype Loss (Metric Learning)

```
Algorithm:
  1. Compute class prototypes:
     For each class c:
       mask_c = (labels == c)
       proto_c = mean(features[mask_c], dim=0)  # [512]
     prototypes = stack([proto_c for c in range(num_classes)])  # [C, 512]
  
  2. Normalize for cosine similarity:
     features_norm = L2_normalize(features, dim=1)  # [B, 512]
     prototypes_norm = L2_normalize(prototypes, dim=1)  # [C, 512]
  
  3. Compute similarity:
     similarity = features_norm @ prototypes_norm.T  # [B, C]
     Each element = cosine(feature_i, proto_c)
  
  4. Apply classification loss on similarities:
     loss_proto = CrossEntropyLoss(similarity, labels)
```

This acts as metric learning that encourages:
- Features of same class to cluster together
- Features of different classes to push apart
- Consistent class prototypes across subjects

### Phase 4: Total Loss and Optimization

```
loss_total = loss_ce + λ₁·loss_coral + λ₂·loss_proto
           = loss_ce + 0.1·loss_coral + 0.5·loss_proto

Parameters to optimize: θ (all model weights)
Optimizer: Adam(θ, lr=1e-4)

Gradient descent:
  θ ← θ - α·∇loss_total(θ)
```

## Numerical Stability Considerations

### Issue 1: Covariance Matrix Conditioning

```python
# Current implementation:
batch_cov = H.T @ H / (B - 1 + 1e-8)

# Improved (optional):
batch_cov = H.T @ H / (B - 1 + 1e-6)
# Add regularization:
batch_cov += ridge_lambda * I  # where I = identity, ridge_lambda=1e-4
```

### Issue 2: Prototype Normalization Edge Case

```python
# If feature is zero:
features_norm = F.normalize(features, p=2, dim=1)  # Creates NaN

# Better:
epsilon = 1e-6
norm = torch.norm(features, p=2, dim=1, keepdim=True).clamp(min=epsilon)
features_norm = features / norm
```

### Issue 3: Empty Classes in Batch

```
Problem: If a class doesn't appear in current batch:
  - proto_c computed from 0 samples → undefined
  - Similarity to non-existent class is meaningless

Solution (current):
  - proto_c = zeros_like(features[0])
  
Better solution:
  - Skip CE loss for missing classes
  - Or use moving average prototypes from previous batches
```

## Memory Requirements

For batch_size=64, feature_dim=512, num_subjects=9:

```
Statistics Storage:
  - mean: [9, 512] × 4 bytes = 18.4 KB
  - cov: [9, 512, 512] × 4 bytes = 9.4 MB
  - Total: ~9.4 MB (negligible)

Gradient Storage:
  - ResNet18: ~200M parameters
  - Gradient: same size = 800 MB

Batch Data:
  - Features: [64, 512] × 4 bytes = 131 KB
  - Covariance computation: [512, 512] = 1 MB
  - Total per batch: ~1.1 MB
```

## Convergence Analysis

### Expected Training Dynamics

```
Epoch 1-10: CORAL loss stays high (statistics not yet meaningful)
           Proto loss helps with class alignment
           CE loss: primary driver of learning

Epoch 10-50: CORAL loss decreases (statistics align better)
            Proto loss provides auxiliary learning signal
            Subject statistics become more stable

Epoch 50+: CORAL loss plateaus
          Fine-tuning of class boundaries
          Subject distributions largely aligned
```

### Hyperparameter Sensitivity

```
Lambda₁ (CORAL weight):
  - 0.0: Only classification + prototype (weaker alignment)
  - 0.1: Good balance (RECOMMENDED)
  - 0.5+: Heavy alignment, may hurt class-specific learning
  
Lambda₂ (Prototype weight):
  - 0.0: No prototype loss (weaker metric learning)
  - 0.2: Weak auxiliary signal
  - 0.5: Good balance (RECOMMENDED)
  - 1.0+: May hurt primary classification task
  
EMA Alpha:
  - 0.5: Fast forgetting, recent batches dominate
  - 0.9: Balanced memory (RECOMMENDED)
  - 0.95+: Long memory, statistics change slowly
```

## Batch Size Effects

CapgMyo dataset example:
```
Total samples per subject: 1280
Batch size options:

batch=32:
  - 40 batches per subject (epoch covers all subjects)
  - Smaller batches → higher variance in statistics
  - More frequent memory updates
  - Probability of all classes in batch: ~80%
  
batch=64: ← RECOMMENDED
  - 20 batches per subject
  - Better statistics estimation
  - Good class balance in batches
  - Faster training (fewer iterations)
  
batch=128:
  - 10 batches per subject
  - Very stable statistics
  - May miss rare classes
  - Slower per-sample training
```

## Extension Points

### Optional 1: Diagonal Covariance

Replace full 512×512 covariance with diagonal:

```python
# Current: O(512²) space
batch_cov = (batch_centered ** 2).mean(dim=0)  # [512] (diagonal)

# Effect: Assumes feature dimensions are independent
# Pros: Much faster, less memory, numerical stability
# Cons: Loses correlation information
```

### Optional 2: Adaptive Loss Weights

```python
# Curriculum learning: gradually increase CORAL weight
current_epoch = epoch
max_epochs = total_epochs
lambda1 = 0.1 * (current_epoch / max_epochs)  # Linear warm-up

# Or based on loss magnitude:
coral_loss_norm = loss_coral / max(1.0, loss_ce)
adaptive_lambda1 = 0.1 * min(1.0, coral_loss_norm)
```

### Optional 3: Temperature Scaling for Prototype Loss

```python
# Soften similarity scores during early training
temperature = 1.0 + (1 - epoch/max_epochs) * 4  # Start at 5, end at 1
softened_similarity = similarity / temperature
loss_proto = CrossEntropyLoss(softened_similarity, labels)
```

### Optional 4: Subject-Specific Prototypes

Instead of batch prototypes, maintain running prototypes per subject:

```python
# In SubjectStatisticsMemory:
self.prototypes = torch.zeros(num_subjects, num_classes, feature_dim)

# In training:
for c in range(num_classes):
  mask = (labels == c)
  if mask.sum() > 0:
    batch_proto = features[mask].mean(dim=0)
    old_proto = self.prototypes[subject_id, c]
    self.prototypes[subject_id, c] = 0.9*old_proto + 0.1*batch_proto
```

## Debugging Checklist

```
□ Check sampling: Verify each batch is from single subject
  - Print subject_ids in training loop
  - Confirm all batch elements have same subject_id

□ Check statistics: Verify SubjectStatisticsMemory updates
  - Print memory.updated[subject_id] progression
  - Print mean/cov changes: ||new_mean - old_mean||

□ Check target distribution: Verify other subjects used
  - Print which subjects have memory.updated=True
  - Verify target_mean differs from batch_mean

□ Check loss computation: All losses should decrease
  - loss_ce should decrease steadily
  - loss_coral should decrease after first epoch
  - loss_proto should stabilize

□ Check gradient: Verify backprop works
  - Print loss.requires_grad (should be True)
  - Check optimizer.param_groups[0]['lr'] is nonzero
```

## Performance Optimization

### Current Implementation Cost

Per batch (64 samples, 512 features):
- Forward pass: ~5ms (ResNet18 backbone)
- CORAL statistics: ~2ms (matrix operations)
- Prototype loss: ~1ms (normalization + MM)
- Backward pass: ~15ms (includes gradient accumulation)
- **Total: ~23ms per batch**

### Optimization Opportunities

1. **Mixed Precision Training**:
   ```python
   # Use float16 for forward, float32 for loss
   with torch.cuda.amp.autocast():
       output, features = model(X_batch)
   # Could reduce time by 30-40%
   ```

2. **Gradient Accumulation**:
   ```python
   # Process 2 batches before update
   # Reduces memory usage, same final gradient
   ```

3. **Covariance Caching**:
   ```python
   # Cache target_cov, recompute only every N batches
   # Trades accuracy for speed
   ```

## References for Further Study

- Deep CORAL (Sun et al., 2016): Domain adaptation via correlation alignment
- Domain Adversarial Neural Networks (Ganin & Lempitsky, 2015): Alternative approach
- Metric Learning (Siamese networks, Triplet loss): Related framework
- Batch Normalization Effects on Domain Adaptation (Li et al., 2018)
