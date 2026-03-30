#!/usr/bin/env python3
"""
Validation script for LOSO CORAL implementation

Tests:
1. SubjectStatisticsMemory: EMA updates and target distribution
2. CorralLoss: Loss computation and gradient flow
3. PrototypeLoss: Prototype generation and similarity
4. SingleSubjectPerBatchSampler: Batch composition
5. CustomDatasetWithSubjectID: Subject ID tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Model.LOSO_CORAL_Trainer import (
    SubjectStatisticsMemory,
    CorralLoss,
    PrototypeLoss,
    SingleSubjectPerBatchSampler,
    CustomDatasetWithSubjectID
)


class TestSubjectStatisticsMemory:
    """Test SubjectStatisticsMemory class"""
    
    def __init__(self):
        self.device = 'cpu'
    
    def test_initialization(self):
        """Test memory initialization"""
        print("\n[TEST] SubjectStatisticsMemory initialization...")
        
        memory = SubjectStatisticsMemory(
            num_subjects=9,
            feature_dim=512,
            ema_alpha=0.9,
            device=self.device
        )
        
        assert memory.mean.shape == (9, 512), f"Expected (9, 512), got {memory.mean.shape}"
        assert memory.cov.shape == (9, 512, 512), f"Expected (9, 512, 512), got {memory.cov.shape}"
        assert memory.updated.sum() == 0, "All subjects should be unupdated initially"
        
        print("✓ Initialization test passed")
    
    def test_ema_update(self):
        """Test EMA update mechanism"""
        print("\n[TEST] SubjectStatisticsMemory EMA update...")
        
        memory = SubjectStatisticsMemory(
            num_subjects=3,
            feature_dim=10,
            ema_alpha=0.9,
            device=self.device
        )
        
        # First update
        features1 = torch.randn(5, 10)
        memory.update(0, features1)
        
        mean1 = memory.mean[0].clone()
        expected_mean1 = features1.mean(dim=0)
        
        assert torch.allclose(mean1, expected_mean1, atol=1e-5), \
            "First update should directly set mean"
        assert memory.updated[0], "Subject 0 should be marked as updated"
        
        # Second update with EMA
        features2 = torch.randn(5, 10)
        memory.update(0, features2)
        
        mean2 = memory.mean[0].clone()
        expected_mean2 = 0.9 * expected_mean1 + 0.1 * features2.mean(dim=0)
        
        assert torch.allclose(mean2, expected_mean2, atol=1e-5), \
            "Second update should use EMA formula"
        
        print("✓ EMA update test passed")
    
    def test_target_distribution(self):
        """Test target distribution retrieval"""
        print("\n[TEST] SubjectStatisticsMemory target distribution...")
        
        memory = SubjectStatisticsMemory(
            num_subjects=3,
            feature_dim=10,
            ema_alpha=0.9,
            device=self.device
        )
        
        # Update all subjects
        for subject_id in range(3):
            features = torch.randn(5, 10)
            memory.update(subject_id, features)
        
        # Get target for subject 0 (exclude current)
        target_mean, target_cov = memory.get_target_distribution(0, exclude_current=True)
        
        assert target_mean.shape == (10,), f"Expected (10,), got {target_mean.shape}"
        assert target_cov.shape == (10, 10), f"Expected (10, 10), got {target_cov.shape}"
        
        # Verify it's average of subjects 1 and 2
        expected_mean = (memory.mean[1] + memory.mean[2]) / 2
        assert torch.allclose(target_mean, expected_mean, atol=1e-5), \
            "Target mean should be average of other subjects"
        
        print("✓ Target distribution test passed")


class TestLossFunctions:
    """Test loss functions"""
    
    def __init__(self):
        self.device = 'cpu'
    
    def test_coral_loss(self):
        """Test CORAL loss computation"""
        print("\n[TEST] CorralLoss computation...")
        
        loss_fn = CorralLoss()
        
        # Test with identical distributions (should be ~0)
        mean = torch.randn(512)
        cov = torch.eye(512) * 0.1
        
        loss = loss_fn(mean, cov, mean, cov)
        assert loss.item() < 1e-5, f"Identical distributions should have near-zero loss, got {loss.item()}"
        
        # Test with different distributions
        mean1 = torch.randn(512)
        cov1 = torch.eye(512) * 0.1
        mean2 = mean1 + torch.randn(512) * 0.5
        cov2 = torch.eye(512) * 0.2
        
        loss = loss_fn(mean1, cov1, mean2, cov2)
        assert loss.item() > 0, "Different distributions should have positive loss"
        assert torch.isfinite(loss), "Loss should be finite"
        
        # Test gradient flow
        loss.backward()
        
        print("✓ CORAL loss test passed")
    
    def test_prototype_loss(self):
        """Test prototype loss computation"""
        print("\n[TEST] PrototypeLoss computation...")
        
        num_classes = 4
        loss_fn = PrototypeLoss(num_classes)
        
        # Create synthetic data
        batch_size = 16
        feature_dim = 512
        features = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        loss = loss_fn(features, labels)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
        
        # Test gradient flow
        loss.backward()
        
        # Test with single-class batch
        features_single = torch.randn(10, feature_dim)
        labels_single = torch.zeros(10, dtype=torch.long)
        
        loss_single = loss_fn(features_single, labels_single)
        assert torch.isfinite(loss_single), "Single class loss should be finite"
        
        print("✓ Prototype loss test passed")


class TestDataHandling:
    """Test dataset and sampler"""
    
    def __init__(self):
        self.device = 'cpu'
    
    def test_custom_dataset_with_subject_id(self):
        """Test CustomDatasetWithSubjectID"""
        print("\n[TEST] CustomDatasetWithSubjectID...")
        
        # Create synthetic data
        # 3 subjects, 20 samples each = 60 total
        X = torch.randn(60, 3, 128, 224)
        Y = torch.eye(4)[torch.randint(0, 4, (60,))]  # 4 classes
        cumulative_sizes = [20, 40, 60]  # Cumulative indices
        
        dataset = CustomDatasetWithSubjectID(X, Y, cumulative_sizes)
        
        assert len(dataset) == 60, f"Expected 60 samples, got {len(dataset)}"
        
        # Check subject IDs
        for idx in range(20):
            sample = dataset[idx]
            assert len(sample) == 3, "Should return (X, Y, subject_id)"
            x, y, subject_id = sample
            assert subject_id == 0, f"Indices 0-19 should be subject 0, got {subject_id}"
        
        for idx in range(20, 40):
            x, y, subject_id = dataset[idx]
            assert subject_id == 1, f"Indices 20-39 should be subject 1, got {subject_id}"
        
        for idx in range(40, 60):
            x, y, subject_id = dataset[idx]
            assert subject_id == 2, f"Indices 40-59 should be subject 2, got {subject_id}"
        
        print("✓ CustomDatasetWithSubjectID test passed")
    
    def test_single_subject_per_batch_sampler(self):
        """Test SingleSubjectPerBatchSampler"""
        print("\n[TEST] SingleSubjectPerBatchSampler...")
        
        batch_size = 8
        cumulative_sizes = [20, 40, 60]  # 3 subjects, 20 samples each
        num_subjects = 3
        
        sampler = SingleSubjectPerBatchSampler(batch_size, cumulative_sizes, num_subjects)
        
        indices = list(sampler.__iter__())
        
        # Check that batches contain only one subject
        batches = []
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i+batch_size]
            if len(batch) == batch_size:  # Only full batches
                batches.append(batch)
                
                # Determine subject for this batch
                subjects = set()
                for idx in batch:
                    if idx < 20:
                        subjects.add(0)
                    elif idx < 40:
                        subjects.add(1)
                    else:
                        subjects.add(2)
                
                assert len(subjects) == 1, f"Batch contains multiple subjects: {subjects}"
        
        print(f"✓ SingleSubjectPerBatchSampler test passed ({len(batches)} batches created)")


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def __init__(self):
        self.device = 'cpu'
    
    def test_full_training_step(self):
        """Test a complete training step"""
        print("\n[TEST] Full training step integration...")
        
        batch_size = 4
        num_classes = 3
        num_subjects = 2
        feature_dim = 32  # Reduced for testing
        
        # Create memory
        memory = SubjectStatisticsMemory(
            num_subjects=num_subjects,
            feature_dim=feature_dim,
            device=self.device
        )
        
        # Create loss functions
        coral_loss = CorralLoss()
        proto_loss = PrototypeLoss(num_classes)
        ce_loss = nn.CrossEntropyLoss()
        
        # Simulate two batches from different subjects
        for subject_id in range(num_subjects):
            # Batch from subject
            features = torch.randn(batch_size, feature_dim, requires_grad=True)
            logits = torch.randn(batch_size, num_classes)
            labels = torch.randint(0, num_classes, (batch_size,))
            labels_onehot = torch.eye(num_classes)[labels]
            
            # Classification loss
            loss_ce = ce_loss(logits, labels)
            
            # CORAL loss
            batch_mean = features.mean(dim=0)
            batch_cov = torch.eye(feature_dim) * 0.1  # Simplified
            target_mean, target_cov = memory.get_target_distribution(subject_id, exclude_current=True)
            loss_coral = coral_loss(batch_mean, batch_cov, target_mean, target_cov)
            
            # Prototype loss
            loss_proto = proto_loss(features, labels)
            
            # Combined loss
            total_loss = loss_ce + 0.1 * loss_coral + 0.5 * loss_proto
            
            # Should be able to backprop
            total_loss.backward()
            
            # Update memory
            memory.update(subject_id, features.detach())
            
            assert torch.isfinite(loss_ce), "CE loss should be finite"
            assert torch.isfinite(loss_coral), "CORAL loss should be finite"
            assert torch.isfinite(loss_proto), "Prototype loss should be finite"
            assert torch.isfinite(total_loss), "Total loss should be finite"
        
        print("✓ Full training step test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("LOSO CORAL Implementation Validation Tests")
    print("=" * 70)
    
    tests = [
        ('SubjectStatisticsMemory', TestSubjectStatisticsMemory()),
        ('Loss Functions', TestLossFunctions()),
        ('Data Handling', TestDataHandling()),
        ('Integration', TestIntegration()),
    ]
    
    failed = []
    
    for test_name, test_class in tests:
        try:
            print(f"\n{'='*70}")
            print(f"Testing: {test_name}")
            print('='*70)
            
            for method_name in dir(test_class):
                if method_name.startswith('test_'):
                    method = getattr(test_class, method_name)
                    print(f"\n  Running {method_name}...")
                    method()
        
        except Exception as e:
            failed.append((test_name, method_name, str(e)))
            print(f"✗ FAILED: {e}")
    
    print(f"\n{'='*70}")
    print("Test Summary")
    print('='*70)
    
    if not failed:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test_name, method_name, error in failed:
            print(f"  - {test_name}.{method_name}: {error}")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
