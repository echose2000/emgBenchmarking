#!/usr/bin/env python3
"""
Quick start script for LOSO CORAL training

Example usage:
    python loso_coral_quickstart.py --dataset capgmyo --epochs 50
"""

import argparse
import sys
from CNN_EMG import main

def run_loso_coral(dataset='capgmyo', epochs=100, batch_size=64, 
                   learning_rate=0.0001, seed=42):
    """
    Run LOSO CORAL training with specified parameters
    
    Args:
        dataset: EMG dataset name (capgmyo, hyser, etc.)
        epochs: Number of training epochs
        batch_size: Batch size (64 recommended)
        learning_rate: Learning rate for Adam optimizer
        seed: Random seed for reproducibility
    """
    
    # Build config arguments for LOSO CORAL
    config_args = {
        'dataset': dataset,
        'model': 'resnet18',
        'domain_generalization': 'CORAL',
        'leave_one_subject_out': True,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seed': seed,
        'pretrain_and_finetune': False,
        'transfer_learning': False,
        'preprocessing': 'spectrogram',
        'gpu': 0
    }
    
    print("=" * 60)
    print("LOSO CORAL Training")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Model: resnet18")
    print(f"Method: LOSO + CORAL + Prototype Loss")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Seed: {seed}")
    print("=" * 60)
    print("\nLoss Weights (hardcoded in LOSO_CORAL_Trainer):")
    print("  - lambda1 (CORAL loss): 0.1")
    print("  - lambda2 (Prototype loss): 0.5")
    print("  - EMA alpha (statistics update): 0.9")
    print("\nBatch Composition:")
    print("  - Each batch: Single subject (LOSO constraint)")
    print("  - Subject statistics: Maintained in SubjectStatisticsMemory")
    print("  - CORAL target: Averaged statistics of other subjects")
    print("=" * 60 + "\n")
    
    # Run training
    main(config_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quick start for LOSO CORAL training'
    )
    
    parser.add_argument('--dataset', type=str, default='capgmyo',
                       help='Dataset name (capgmyo, hyser, myoarmbanddataset, etc.)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate for Adam optimizer')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    try:
        run_loso_coral(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
