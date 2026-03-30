"""
LOSO CORAL with Subject Memory
Implements subject-level statistics memory, CORAL loss, and prototype loss
for cross-subject EMG classification in LOSO setting.
"""

from .Model_Trainer import Model_Trainer
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm
import Model.VisualTransformer as VisualTransformer
import Model.ml_metrics_utils as ml_utils
import numpy as np
from torch.utils.data import DataLoader, Sampler, Dataset
import multiprocessing
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import torch.autograd as autograd
import torch.nn.functional as F
import math
import copy
import random
import os


class ResNet18WithFeatures(nn.Module):
    """ResNet18 backbone with explicit feature extraction"""
    def __init__(self, model_name, num_classes):
        super(ResNet18WithFeatures, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        # Extract the feature extractor part, excluding classification layer
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])  
        # Classifier layer
        self.classifier = nn.Linear(self.resnet.num_features, num_classes)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x).view(x.size(0), -1)
        # Get predictions
        predictions = self.classifier(features)
        return predictions, features


class SubjectStatisticsMemory:
    """
    Maintains running statistics for each subject: mean and covariance of features.
    Uses exponential moving average (EMA) for updates.
    """
    def __init__(self, num_subjects, feature_dim, ema_alpha=0.9, device='cuda:0'):
        """
        Args:
            num_subjects: number of subjects in training set
            feature_dim: dimension of feature vectors (e.g., 512)
            ema_alpha: EMA weight for old statistics (e.g., 0.9)
            device: device to store statistics
        """
        self.num_subjects = num_subjects
        self.feature_dim = feature_dim
        self.ema_alpha = ema_alpha
        self.device = device
        
        # Initialize statistics for each subject
        # mean: [num_subjects, feature_dim]
        # cov: [num_subjects, feature_dim, feature_dim] or [num_subjects, feature_dim] for diagonal
        self.mean = torch.zeros(num_subjects, feature_dim, device=device)
        self.cov = torch.zeros(num_subjects, feature_dim, feature_dim, device=device)
        
        # Track if subject has been updated
        self.updated = torch.zeros(num_subjects, dtype=torch.bool, device=device)
    
    def update(self, subject_id, batch_features):
        """
        Update statistics for a subject using EMA.
        
        Args:
            subject_id: index of the subject (0 to num_subjects-1)
            batch_features: [batch_size, feature_dim] tensor of features
        """
        # Compute batch mean and covariance
        batch_mean = batch_features.mean(dim=0)  # [feature_dim]
        batch_centered = batch_features - batch_mean  # [batch_size, feature_dim]
        batch_cov = torch.mm(batch_centered.t(), batch_centered) / (len(batch_features) - 1 + 1e-8)  # [feature_dim, feature_dim]
        
        if self.updated[subject_id]:
            # Update using EMA
            self.mean[subject_id] = self.ema_alpha * self.mean[subject_id] + (1 - self.ema_alpha) * batch_mean
            self.cov[subject_id] = self.ema_alpha * self.cov[subject_id] + (1 - self.ema_alpha) * batch_cov
        else:
            # First update: initialize
            self.mean[subject_id] = batch_mean
            self.cov[subject_id] = batch_cov
            self.updated[subject_id] = True
    
    def get_target_distribution(self, current_subject_id, exclude_current=True):
        """
        Get the target distribution (mean and covariance) for a subject.
        Averages statistics of all other subjects if exclude_current=True.
        
        Args:
            current_subject_id: the subject we're currently training
            exclude_current: whether to exclude current subject from averaging
        
        Returns:
            target_mean: [feature_dim]
            target_cov: [feature_dim, feature_dim]
        """
        if exclude_current:
            # Average all subjects except current one
            mask = torch.ones(self.num_subjects, dtype=torch.bool, device=self.device)
            mask[current_subject_id] = False
            
            # Only average subjects that have been updated
            valid_mask = mask & self.updated
            valid_count = valid_mask.sum().item()
            
            if valid_count == 0:
                # No valid subjects to average from, return zeros
                return self.mean[current_subject_id].clone(), self.cov[current_subject_id].clone()
            
            target_mean = self.mean[valid_mask].mean(dim=0)
            target_cov = self.cov[valid_mask].mean(dim=0)
        else:
            # Average all subjects
            valid_mask = self.updated
            valid_count = valid_mask.sum().item()
            
            if valid_count == 0:
                return torch.zeros(self.feature_dim, device=self.device), \
                       torch.zeros(self.feature_dim, self.feature_dim, device=self.device)
            
            target_mean = self.mean[valid_mask].mean(dim=0)
            target_cov = self.cov[valid_mask].mean(dim=0)
        
        return target_mean, target_cov


class CorralLoss(nn.Module):
    """
    Correlation Alignment Loss (Deep CORAL) for feature distribution alignment.
    Computes Frobenius norm between covariance matrices.
    """
    def __init__(self):
        super(CorralLoss, self).__init__()
    
    def forward(self, source_mean, source_cov, target_mean, target_cov):
        """
        Compute CORAL loss between source and target distributions.
        
        Args:
            source_mean: [feature_dim] source distribution mean
            source_cov: [feature_dim, feature_dim] source distribution covariance
            target_mean: [feature_dim] target distribution mean
            target_cov: [feature_dim, feature_dim] target distribution covariance
        
        Returns:
            scalar loss value
        """
        # Mean difference
        mean_diff = torch.pow(source_mean - target_mean, 2).mean()
        
        # Covariance difference (Frobenius norm)
        cov_diff = torch.pow(source_cov - target_cov, 2).mean()
        
        return mean_diff + cov_diff


class PrototypeLoss(nn.Module):
    """
    Prototype-based metric learning loss.
    Computes cosine similarity between samples and class prototypes.
    """
    def __init__(self, num_classes):
        super(PrototypeLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, features, labels):
        """
        Compute prototype loss.
        
        Args:
            features: [batch_size, feature_dim] feature vectors
            labels: [batch_size] class labels (long tensor)
        
        Returns:
            scalar loss value
        """
        # Compute class prototypes (mean feature per class)
        prototypes = []
        for c in range(self.num_classes):
            class_mask = labels == c
            if class_mask.sum() > 0:
                prototype = features[class_mask].mean(dim=0)
                prototypes.append(prototype)
            else:
                # If class not in batch, use zero prototype
                prototypes.append(torch.zeros_like(features[0]))
        
        prototypes = torch.stack(prototypes)  # [num_classes, feature_dim]
        
        # Normalize features and prototypes for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)  # [batch_size, feature_dim]
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)  # [num_classes, feature_dim]
        
        # Compute cosine similarity: [batch_size, num_classes]
        similarity = torch.mm(features_norm, prototypes_norm.t())
        
        # Apply cross-entropy loss
        return self.ce_loss(similarity, labels)


class CustomDatasetWithSubjectID(Dataset):
    """
    Custom dataset that tracks which subject each sample belongs to.
    """
    def __init__(self, X, Y, cumulative_sizes, transform=None):
        """
        Args:
            X: input features
            Y: labels
            cumulative_sizes: cumulative dataset sizes for each subject
            transform: optional transformation
        """
        self.X = X
        self.Y = Y
        self.transform = transform
        self.cumulative_sizes = cumulative_sizes
        
        # Build subject_id for each sample
        self.subject_ids = []
        start = 0
        for subject_id, end in enumerate(cumulative_sizes):
            self.subject_ids.extend([subject_id] * (end - start))
            start = end
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        subject_id = self.subject_ids[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y, subject_id


class SingleSubjectPerBatchSampler(Sampler):
    """
    Sampler that ensures each batch contains samples from only one subject.
    Useful for LOSO setting where we want to maintain subject-level statistics.
    """
    def __init__(self, batch_size, cumulative_sizes, num_subjects):
        """
        Args:
            batch_size: size of each batch
            cumulative_sizes: cumulative dataset sizes for each subject
            num_subjects: total number of subjects in training set
        """
        super(SingleSubjectPerBatchSampler, self).__init__()
        self.batch_size = batch_size
        self.cumulative_sizes = cumulative_sizes
        self.num_subjects = num_subjects
        
        # Calculate indices for each subject
        self.subject_indices = []
        start = 0
        for end in cumulative_sizes:
            self.subject_indices.append(list(range(start, end)))
            start = end
        
        self.length = len(list(self.__iter__()))
    
    def __iter__(self):
        # Iterate through subjects and yield batches from each subject
        final_indices = []
        
        for subject_id in range(self.num_subjects):
            subject_data = self.subject_indices[subject_id].copy()
            random.shuffle(subject_data)
            
            # Create batches from this subject's data
            for i in range(0, len(subject_data), self.batch_size):
                batch = subject_data[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only include full batches
                    final_indices.extend(batch)
        
        return iter(final_indices)
    
    def __len__(self):
        return self.length
    
    def get_subject_for_index(self, idx):
        """Helper function to determine which subject a sample index belongs to"""
        for subject_id, subject_indices in enumerate(self.subject_indices):
            if idx in subject_indices:
                return subject_id
        return -1


class LOSO_CORAL_Trainer(Model_Trainer):
    """
    Training class for LOSO+CORAL with subject-level statistics memory and prototype loss.
    """
    
    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)
        
        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Subject statistics memory will be initialized in setup_model
        self.subject_stats_memory = None
        
        # Loss weights (can be modified via command line arguments)
        self.lambda1 = self.args.lambda_coral  # CORAL loss weight
        self.lambda2 = self.args.lambda_prototype  # Prototype loss weight
    
    def setup_model(self):
        """Set up all model parameters for the run."""
        super().set_pretrain_path()
        self.set_model()
        self.set_optimizer()
        self.set_param_requires_grad()
        super().set_resize_transform()
        self.set_loaders()  # Custom loaders for LOSO
        self.set_criterion()
        
        # Print loss weights
        print(f"\n{'='*70}")
        print(f"LOSO CORAL Loss Weights Configuration")
        print(f"{'='*70}")
        print(f"CORAL loss weight (lambda1): {self.lambda1}")
        print(f"Prototype loss weight (lambda2): {self.lambda2}")
        print(f"Total loss = CE + {self.lambda1} × CORAL + {self.lambda2} × Prototype")
        print(f"{'='*70}\n")
        
        super().start_pretrain_run()
        super().set_model_to_device()
        super().set_testrun_foldername()
        super().set_gesture_labels()
        super().plot_images()
    
    def set_model(self):
        """Set up ResNet18 with features"""
        self.model = ResNet18WithFeatures(model_name=self.model_name, num_classes=self.num_gestures)
        
        # Calculate the number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total number of parameters: {total_params}')
        
        # Initialize subject statistics memory
        # Feature dimension is 512 for ResNet18
        num_subjects = self.utils.num_subjects - 1 if self.args.leave_one_subject_out else self.utils.num_subjects
        self.subject_stats_memory = SubjectStatisticsMemory(
            num_subjects=num_subjects,
            feature_dim=512,
            ema_alpha=0.9,
            device=self.device
        )
    
    def set_param_requires_grad(self):
        """Set which parameters require grad"""
        for name, param in self.model.named_parameters():
            param.requires_grad = True
    
    def set_optimizer(self):
        """Set Adam optimizer"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    
    def set_criterion(self):
        """Set loss functions"""
        self.cross_entropy = nn.CrossEntropyLoss()
        self.coral_loss = CorralLoss()
        self.prototype_loss = PrototypeLoss(self.num_gestures)
    
    def set_loaders(self):
        """Create data loaders with SingleSubjectPerBatchSampler for LOSO"""
        # Note: We'll create datasets manually to use CustomDatasetWithSubjectID
        train_dataset = CustomDatasetWithSubjectID(
            self.X.train, 
            self.Y.train,
            self.X.cumulative_sizes,
            transform=self.resize_transform
        )
        
        val_dataset = self.CustomDataset(
            self.X.validation, 
            self.Y.validation, 
            transform=self.resize_transform
        )
        
        test_dataset = self.CustomDataset(
            self.X.test, 
            self.Y.test, 
            transform=self.resize_transform
        )
        
        # Use SingleSubjectPerBatchSampler to ensure one subject per batch
        sampler = SingleSubjectPerBatchSampler(
            batch_size=self.args.batch_size,
            cumulative_sizes=self.X.cumulative_sizes,
            num_subjects=self.utils.num_subjects - 1 if self.args.leave_one_subject_out else self.utils.num_subjects
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=multiprocessing.cpu_count()//8,
            worker_init_fn=self.utils.seed_worker,
            pin_memory=True,
            sampler=sampler,
            drop_last=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            num_workers=multiprocessing.cpu_count()//8,
            worker_init_fn=self.utils.seed_worker,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            num_workers=multiprocessing.cpu_count()//8,
            worker_init_fn=self.utils.seed_worker,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
        
        self.sampler = sampler
        self.train_dataset = train_dataset
    
    
    def model_loop(self):
        """Main training loop"""
        # Get metrics
        if self.args.pretrain_and_finetune:
            training_metrics, validation_metrics, testing_metrics = super().get_metrics()
        else: 
            training_metrics, validation_metrics = super().get_metrics(testing=False)
        
        # Train and Validation Loop 
        self.train_and_validate(training_metrics, validation_metrics)
        
        # 保存混淆矩阵和预测结果
        self.save_and_print_results()
        
        # Few-shot adaptation (if enabled)
        if self.args.adapt_ratio > 0.0:
            self.adaptive_test_with_few_shot()
        
        # Finetune Loop 
        if self.args.pretrain_and_finetune:
            self.pretrain_and_finetune(testing_metrics)
    
    def train_and_validate(self, training_metrics, validation_metrics):
        """Train and validation loop with CORAL and prototype losses"""
        
        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            
            self.model.train()
            train_loss = 0.0
            
            # Reset training metrics at the start of each epoch
            for train_metric in training_metrics:
                train_metric.reset()
            
            outputs_train_all = []
            ground_truth_train_all = []
            batch_no = 0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False) as t:
                
                for batch_data in t:
                    # Unpack batch with subject IDs
                    if len(batch_data) == 3:
                        X_batch, Y_batch, subject_ids = batch_data
                        # All samples in batch should be from same subject
                        current_subject_id = subject_ids[0].item()
                    else:
                        # Fallback if dataset doesn't return subject ID
                        X_batch, Y_batch = batch_data
                        current_subject_id = batch_no % (self.utils.num_subjects - 1)
                    
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32)
                    Y_batch_long = torch.argmax(Y_batch, dim=1)
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    output, features = self.model(X_batch)  # output: [64, num_classes], features: [64, 512]
                    
                    if isinstance(output, dict):
                        output = output['logits']
                    
                    # === Compute Classification Loss ===
                    loss_ce = self.cross_entropy(output, Y_batch_long)
                    
                    # Compute CORAL Loss ===
                    # Estimate batch statistics and CORAL loss
                    batch_mean = features.mean(dim=0)  # [512]
                    batch_centered = features - batch_mean
                    batch_cov = torch.mm(batch_centered.t(), batch_centered) / (len(features) - 1 + 1e-8)  # [512, 512]
                    
                    # Get target distribution from other subjects
                    target_mean, target_cov = self.subject_stats_memory.get_target_distribution(
                        current_subject_id, exclude_current=True
                    )
                    
                    # Compute CORAL loss
                    loss_coral = self.coral_loss(batch_mean, batch_cov, target_mean, target_cov)
                    
                    # === Compute Prototype Loss ===
                    loss_prototype = self.prototype_loss(features, Y_batch_long)
                    
                    # === Total Loss ===
                    loss = loss_ce + self.lambda1 * loss_coral + self.lambda2 * loss_prototype
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update subject statistics memory
                    self.subject_stats_memory.update(current_subject_id, features.detach())
                    
                    outputs_train_all.append(output)
                    ground_truth_train_all.append(Y_batch_long)
                    
                    train_loss += loss.item()
                    
                    for train_metric in training_metrics:
                        if train_metric.name not in ["Macro_AUROC", "Macro_AUPRC"]:
                            train_metric(output, Y_batch_long)
                    
                    micro_accuracy_metric = next(metric for metric in training_metrics if metric.name == "Micro_Accuracy")
                    if t.n % 10 == 0:
                        t.set_postfix({
                            "Total Loss": loss.item(),
                            "CE Loss": loss_ce.item(),
                            "CORAL Loss": loss_coral.item(),
                            "Proto Loss": loss_prototype.item(),
                            "Batch Acc": micro_accuracy_metric.compute().item()
                        })
                    
                    batch_no += 1
            
            # Concatenate all outputs and labels
            outputs_train_all = torch.cat(outputs_train_all, dim=0).to(self.device)
            ground_truth_train_all = torch.cat(ground_truth_train_all, dim=0).to(self.device)
            
            # Compute AUROC and AUPRC
            train_macro_auroc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUROC")
            train_macro_auprc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUPRC")
            
            train_macro_auroc_metric(outputs_train_all, ground_truth_train_all)
            train_macro_auprc_metric(outputs_train_all, ground_truth_train_all)
            
            # === Validation Phase ===
            self.model.eval()
            val_loss = 0.0
            
            for val_metric in validation_metrics:
                val_metric.reset()
            
            all_val_outputs = []
            all_val_labels = []
            
            with torch.no_grad():
                for X_batch, Y_batch in tqdm(self.val_loader, desc="Validation", leave=False):
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32)
                    Y_batch_long = torch.argmax(Y_batch, dim=1)
                    
                    output, features = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    
                    loss = self.cross_entropy(output, Y_batch_long)
                    val_loss += loss.item()
                    
                    all_val_outputs.append(output)
                    all_val_labels.append(Y_batch_long)
                    
                    for val_metric in validation_metrics:
                        if val_metric.name not in ["Macro_AUROC", "Macro_AUPRC"]:
                            val_metric(output, Y_batch_long)
            
            all_val_outputs = torch.cat(all_val_outputs, dim=0).to(self.device)
            all_val_labels = torch.cat(all_val_labels, dim=0).to(self.device)
            
            val_macro_auroc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUROC")
            val_macro_auprc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUPRC")
            
            val_macro_auroc_metric(all_val_outputs, all_val_labels)
            val_macro_auprc_metric(all_val_outputs, all_val_labels)
            
            # Log metrics
            avg_train_loss = train_loss / batch_no
            avg_val_loss = val_loss / len(self.val_loader)
            
            if self.run:
                self.run.log({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                })
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    def pretrain_and_finetune(self, testing_metrics):
        """Finetune on test set after pretraining"""
        ml_utils.evaluate_model_on_test_set(
            self.model, self.test_loader, self.device, 
            self.num_gestures, self.cross_entropy, self.args, testing_metrics
        )
        
        if not self.args.force_regression:
            self.print_classification_metrics()
    
    def print_classification_metrics(self):
        """Print confusion matrices and classification reports"""
        # Test set metrics
        self.model.eval()
        with torch.no_grad():
            test_predictions = []
            for X_batch, Y_batch in tqdm(self.test_loader, desc="Test", leave=False):
                X_batch = X_batch.to(self.device).to(torch.float32)
                output, _ = self.model(X_batch)
                if isinstance(output, dict):
                    output = output['logits']
                preds = np.argmax(output.cpu().detach().numpy(), axis=1)
                test_predictions.extend(preds)
        
        true_labels = np.argmax(self.Y.test.cpu().detach().numpy(), axis=1)
        test_predictions = np.array(test_predictions)
        
        conf_matrix = confusion_matrix(true_labels, test_predictions)
        print("Test Confusion Matrix:")
        print(conf_matrix)
        print("\nTest Classification Report:")
        print(classification_report(true_labels, test_predictions))
        
        self.utils.plot_confusion_matrix(
            true_labels, test_predictions, self.gesture_labels,
            self.testrun_foldername, self.args, self.formatted_datetime, 'test'
        )
        
        torch.cuda.empty_cache()
    
    def save_and_print_results(self):
        """
        Save and print classification results including:
        - Separate pkl files for test/validation/train sets
        - Separate confusion matrix files
        - All in a single 'result' directory per subject
        """
        import pickle
        import os
        
        # 创建result目录
        result_dir = f'{self.testrun_foldername}result_subject_{self.args.leftout_subject}/'
        os.makedirs(result_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"保存结果到: {result_dir}")
        print("="*70)
        
        self.model.eval()
        
        # ===== TEST SET =====
        print("\n[TEST SET]")
        with torch.no_grad():
            test_predictions = []
            test_features = []
            for batch_data in tqdm(self.test_loader, desc="测试集预测", leave=False):
                if len(batch_data) == 3:
                    X_batch, Y_batch, _ = batch_data
                else:
                    X_batch, Y_batch = batch_data
                
                X_batch = X_batch.to(self.device).to(torch.float32)
                output, features = self.model(X_batch)
                
                if isinstance(output, dict):
                    output = output['logits']
                
                preds = np.argmax(output.cpu().detach().numpy(), axis=1)
                test_predictions.extend(preds)
                test_features.extend(features.cpu().detach().numpy())
        
        test_predictions = np.array(test_predictions)
        test_features = np.array(test_features)
        true_test_labels = np.argmax(self.Y.test.cpu().detach().numpy(), axis=1)
        test_conf_matrix = confusion_matrix(true_test_labels, test_predictions)
        test_acc = (test_predictions == true_test_labels).sum() / len(true_test_labels)
        
        print(f"测试集准确率: {test_acc:.4f}")
        print("测试集混淆矩阵:")
        print(test_conf_matrix)
        print("\n测试集分类报告:")
        print(classification_report(true_test_labels, test_predictions))
        
        # 保存测试集pkl
        test_pkl = {
            'predictions': test_predictions,
            'labels': true_test_labels,
            'features': test_features,
            'confusion_matrix': test_conf_matrix,
            'accuracy': test_acc,
            'gesture_labels': self.gesture_labels,
        }
        test_pkl_file = f'{result_dir}test_results.pkl'
        with open(test_pkl_file, 'wb') as f:
            pickle.dump(test_pkl, f)
        print(f"✓ 测试集pkl已保存: {test_pkl_file}")
        
        # 保存测试集混淆矩阵图
        self.utils.plot_confusion_matrix(
            true_test_labels, test_predictions, self.gesture_labels,
            result_dir, self.args, 'test', 'test_confusion_matrix'
        )
        
        # ===== VALIDATION SET =====
        print("\n[VALIDATION SET]")
        with torch.no_grad():
            val_predictions = []
            val_features = []
            for batch_data in tqdm(self.val_loader, desc="验证集预测", leave=False):
                if len(batch_data) == 3:
                    X_batch, Y_batch, _ = batch_data
                else:
                    X_batch, Y_batch = batch_data
                
                X_batch = X_batch.to(self.device).to(torch.float32)
                output, features = self.model(X_batch)
                
                if isinstance(output, dict):
                    output = output['logits']
                
                preds = np.argmax(output.cpu().detach().numpy(), axis=1)
                val_predictions.extend(preds)
                val_features.extend(features.cpu().detach().numpy())
        
        val_predictions = np.array(val_predictions)
        val_features = np.array(val_features)
        true_val_labels = np.argmax(self.Y.validation.cpu().detach().numpy(), axis=1)
        val_conf_matrix = confusion_matrix(true_val_labels, val_predictions)
        val_acc = (val_predictions == true_val_labels).sum() / len(true_val_labels)
        
        print(f"验证集准确率: {val_acc:.4f}")
        print("验证集混淆矩阵:")
        print(val_conf_matrix)
        print("\n验证集分类报告:")
        print(classification_report(true_val_labels, val_predictions))
        
        # 保存验证集pkl
        val_pkl = {
            'predictions': val_predictions,
            'labels': true_val_labels,
            'features': val_features,
            'confusion_matrix': val_conf_matrix,
            'accuracy': val_acc,
            'gesture_labels': self.gesture_labels,
        }
        val_pkl_file = f'{result_dir}validation_results.pkl'
        with open(val_pkl_file, 'wb') as f:
            pickle.dump(val_pkl, f)
        print(f"✓ 验证集pkl已保存: {val_pkl_file}")
        
        # 保存验证集混淆矩阵图
        self.utils.plot_confusion_matrix(
            true_val_labels, val_predictions, self.gesture_labels,
            result_dir, self.args, 'validation', 'validation_confusion_matrix'
        )
        
        # ===== TRAINING SET =====
        print("\n[TRAINING SET]")
        self.train_loader_unshuffled = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=multiprocessing.cpu_count()//8,
            worker_init_fn=self.utils.seed_worker,
            pin_memory=True,
            drop_last=False
        )
        
        with torch.no_grad():
            train_predictions = []
            train_features = []
            for batch_data in tqdm(self.train_loader_unshuffled, desc="训练集预测", leave=False):
                if len(batch_data) == 3:
                    X_batch, Y_batch, _ = batch_data
                else:
                    X_batch, Y_batch = batch_data
                
                X_batch = X_batch.to(self.device).to(torch.float32)
                output, features = self.model(X_batch)
                
                if isinstance(output, dict):
                    output = output['logits']
                
                preds = np.argmax(output.cpu().detach().numpy(), axis=1)
                train_predictions.extend(preds)
                train_features.extend(features.cpu().detach().numpy())
        
        train_predictions = np.array(train_predictions)
        train_features = np.array(train_features)
        true_train_labels = np.argmax(self.Y.train.cpu().detach().numpy(), axis=1)
        train_conf_matrix = confusion_matrix(true_train_labels, train_predictions)
        train_acc = (train_predictions == true_train_labels).sum() / len(true_train_labels)
        
        print(f"训练集准确率: {train_acc:.4f}")
        print("训练集混淆矩阵:")
        print(train_conf_matrix)
        
        # 保存训练集pkl
        train_pkl = {
            'predictions': train_predictions,
            'labels': true_train_labels,
            'features': train_features,
            'confusion_matrix': train_conf_matrix,
            'accuracy': train_acc,
            'gesture_labels': self.gesture_labels,
        }
        train_pkl_file = f'{result_dir}train_results.pkl'
        with open(train_pkl_file, 'wb') as f:
            pickle.dump(train_pkl, f)
        print(f"✓ 训练集pkl已保存: {train_pkl_file}")
        
        # 保存训练集混淆矩阵图
        self.utils.plot_confusion_matrix(
            true_train_labels, train_predictions, self.gesture_labels,
            result_dir, self.args, 'train', 'train_confusion_matrix'
        )
        
        # ===== 总结报告 =====
        summary_txt = f'{result_dir}summary.txt'
        with open(summary_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"LOSO CORAL 训练结果总结 - Subject {self.args.leftout_subject}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"数据集: {self.args.dataset}\n")
            f.write(f"模型: {self.model_name}\n")
            f.write(f"训练方法: LOSO + CORAL + Prototype Loss\n")
            f.write(f"随机种子: {self.args.seed}\n")
            f.write(f"训练轮数: {self.num_epochs}\n")
            f.write(f"批大小: {self.args.batch_size}\n\n")
            
            f.write("="*70 + "\n")
            f.write("准确率统计\n")
            f.write("="*70 + "\n")
            f.write(f"训练集准确率: {train_acc:.4f}\n")
            f.write(f"验证集准确率: {val_acc:.4f}\n")
            f.write(f"测试集准确率: {test_acc:.4f}\n\n")
            
            f.write("="*70 + "\n")
            f.write("测试集详细报告\n")
            f.write("="*70 + "\n")
            f.write("混淆矩阵:\n")
            f.write(str(test_conf_matrix) + "\n\n")
            f.write(classification_report(true_test_labels, test_predictions) + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("验证集详细报告\n")
            f.write("="*70 + "\n")
            f.write("混淆矩阵:\n")
            f.write(str(val_conf_matrix) + "\n\n")
            f.write(classification_report(true_val_labels, val_predictions) + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("训练集详细报告\n")
            f.write("="*70 + "\n")
            f.write("混淆矩阵:\n")
            f.write(str(train_conf_matrix) + "\n\n")
        
        print(f"✓ 总结报告已保存: {summary_txt}")
        
        print("\n" + "="*70)
        print(f"✓ 所有结果已保存到: {result_dir}")
        print("="*70)
        print(f"\n目录结构:")
        print(f"  {result_dir}")
        print(f"  ├─ train_results.pkl")
        print(f"  ├─ validation_results.pkl")
        print(f"  ├─ test_results.pkl")
        print(f"  ├─ train_confusion_matrix.png")
        print(f"  ├─ validation_confusion_matrix.png")
        print(f"  ├─ test_confusion_matrix.png")
        print(f"  └─ summary.txt")
        
        torch.cuda.empty_cache()
    
    def adaptive_test_with_few_shot(self):
        """
        Few-shot + Transductive Domain Adaptation for Subject 0 (test subject).
        
        Key steps:
        1. Sample adapt_ratio proportion from validation and test sets
        2. Ensure equal size by taking minimum
        3. Estimate Subject 0 distribution using support set
        4. Align test features using computed statistics
        
        Returns:
            dict with adaptation results
        """
        import pickle
        
        if self.args.adapt_ratio <= 0.0:
            print("\n[ADAPT_RATIO = 0.0] Few-shot adaptation disabled.")
            return None
        
        print("\n" + "="*70)
        print(f"Few-shot + Transductive Domain Adaptation")
        print(f"Adaptation Ratio: {self.args.adapt_ratio}")
        print("="*70)
        
        # ===== Step 1: Compute training set distribution =====
        print("\n[Step 1] Computing training set distribution (Subject 1-9)...")
        self.model.eval()
        
        train_features_all = []
        with torch.no_grad():
            for batch_data in tqdm(self.train_loader, desc="Training features", leave=False):
                if len(batch_data) == 3:
                    X_batch, _, _ = batch_data
                else:
                    X_batch, _ = batch_data
                
                X_batch = X_batch.to(self.device).to(torch.float32)
                _, features = self.model(X_batch)
                train_features_all.append(features.cpu().detach().numpy())
        
        train_features_all = np.concatenate(train_features_all, axis=0)
        mean_train = train_features_all.mean(axis=0)  # [512]
        var_train = np.var(train_features_all, axis=0)  # [512] diagonal variance
        
        print(f"  Train features shape: {train_features_all.shape}")
        print(f"  Mean shape: {mean_train.shape}")
        print(f"  Var shape: {var_train.shape}")
        
        # ===== Step 2: Sample support set from val and test =====
        print(f"\n[Step 2] Building support set with adapt_ratio={self.args.adapt_ratio}...")
        
        num_val = len(self.Y.validation)
        num_test = len(self.Y.test)
        
        val_support_size = max(1, int(num_val * self.args.adapt_ratio))
        test_support_size = max(1, int(num_test * self.args.adapt_ratio))
        
        # Ensure equal sizes
        support_size = min(val_support_size, test_support_size)
        
        print(f"  Val set size: {num_val}, support_size: {val_support_size}")
        print(f"  Test set size: {num_test}, support_size: {test_support_size}")
        print(f"  Final support_size (min): {support_size}")
        
        # Random sampling
        np.random.seed(self.args.seed)
        val_indices_all = np.arange(num_val)
        test_indices_all = np.arange(num_test)
        
        val_support_indices = np.random.choice(val_indices_all, size=support_size, replace=False)
        test_support_indices = np.random.choice(test_indices_all, size=support_size, replace=False)
        
        print(f"  Sampled {support_size} from validation (indices: {val_support_indices[:5]}...)")
        print(f"  Sampled {support_size} from test (indices: {test_support_indices[:5]}...)")
        
        # ===== Step 3: Extract features from support set =====
        print(f"\n[Step 3] Extracting support set features...")
        
        support_features = []
        
        # Process validation support using DataLoader
        val_support_dataset = self.CustomDataset(
            self.X.validation[val_support_indices], 
            self.Y.validation[val_support_indices], 
            transform=self.resize_transform
        )
        val_support_loader = DataLoader(
            val_support_dataset,
            batch_size=32,
            num_workers=0,
            pin_memory=True,
            shuffle=False
        )
        
        with torch.no_grad():
            for X_batch, _ in val_support_loader:
                X_batch = X_batch.to(self.device).to(torch.float32)
                _, feat = self.model(X_batch)
                support_features.append(feat.cpu().detach().numpy())
        
        # Process test support using DataLoader
        test_support_dataset = self.CustomDataset(
            self.X.test[test_support_indices], 
            self.Y.test[test_support_indices], 
            transform=self.resize_transform
        )
        test_support_loader = DataLoader(
            test_support_dataset,
            batch_size=32,
            num_workers=0,
            pin_memory=True,
            shuffle=False
        )
        
        with torch.no_grad():
            for X_batch, _ in test_support_loader:
                X_batch = X_batch.to(self.device).to(torch.float32)
                _, feat = self.model(X_batch)
                support_features.append(feat.cpu().detach().numpy())
        
        support_features = np.concatenate(support_features, axis=0)
        
        # ===== Step 4: Estimate Subject 0 distribution =====
        print(f"\n[Step 4] Computing Subject 0 distribution from support set...")
        
        mean_0 = support_features.mean(axis=0)  # [512]
        var_0 = np.var(support_features, axis=0)  # [512]
        
        print(f"  Support set features shape: {support_features.shape}")
        print(f"  Subject 0 mean norm: {np.linalg.norm(mean_0):.4f}")
        print(f"  Subject 0 var mean: {var_0.mean():.6f}")
        
        # ===== Step 5: Test with feature alignment =====
        print(f"\n[Step 5] Testing with feature alignment...")
        
        # Create test set excluding test_support_indices
        test_indices_remain = np.setdiff1d(test_indices_all, test_support_indices)
        
        print(f"  Original test set size: {num_test}")
        print(f"  Test support removed: {support_size}")
        print(f"  Remaining test set size: {len(test_indices_remain)}")
        
        # Create data loader for remaining test samples
        test_remain_dataset = self.CustomDataset(
            self.X.test[test_indices_remain], 
            self.Y.test[test_indices_remain], 
            transform=self.resize_transform
        )
        test_remain_loader = DataLoader(
            test_remain_dataset,
            batch_size=32,
            num_workers=0,
            pin_memory=True,
            shuffle=False
        )
        
        test_predictions = []
        test_features_aligned = []
        test_true_labels = []
        
        eps = 1e-5
        
        with torch.no_grad():
            batch_idx = 0
            for X_batch, Y_batch in tqdm(test_remain_loader, desc="Testing with alignment", leave=False):
                X_batch = X_batch.to(self.device).to(torch.float32)
                Y_batch = Y_batch.to(self.device).to(torch.float32)
                Y_batch_long = torch.argmax(Y_batch, dim=1)
                
                # Forward to get features
                output, feat = self.model(X_batch)
                
                if isinstance(output, dict):
                    output = output['logits']
                
                # Feature alignment for each sample in batch
                feat_np = feat.cpu().detach().numpy()  # [batch_size, 512]
                
                for i in range(len(feat_np)):
                    feat_sample = feat_np[i]  # [512]
                    
                    # Align: (feat - mean_0) / sqrt(var_0 + eps) * sqrt(var_train + eps) + mean_train
                    feat_centered = feat_sample - mean_0
                    feat_normalized = feat_centered / np.sqrt(var_0 + eps)
                    feat_aligned = feat_normalized * np.sqrt(var_train + eps) + mean_train
                    
                    # Convert back to tensor and pass through classifier
                    feat_aligned_tensor = torch.tensor(feat_aligned, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits_aligned = self.model.classifier(feat_aligned_tensor)
                    
                    pred = np.argmax(logits_aligned.cpu().detach().numpy()[0])
                    true_label = Y_batch_long[i].item()
                    
                    test_predictions.append(pred)
                    test_features_aligned.append(feat_aligned)
                    test_true_labels.append(true_label)
                
                batch_idx += 1
        
        test_predictions = np.array(test_predictions)
        test_features_aligned = np.array(test_features_aligned)
        test_true_labels = np.array(test_true_labels)
        
        # ===== Step 6: Compute metrics =====
        print(f"\n[Step 6] Computing metrics...")
        
        test_acc_adapted = (test_predictions == test_true_labels).sum() / len(test_true_labels)
        test_conf_matrix = confusion_matrix(test_true_labels, test_predictions)
        
        print(f"\n{'='*70}")
        print(f"Few-shot Adaptation Results (Subject {self.args.leftout_subject})")
        print(f"{'='*70}")
        print(f"Support set size: {support_size * 2} (val: {support_size}, test: {support_size})")
        print(f"Test samples evaluated: {len(test_true_labels)}")
        print(f"Test Accuracy (with alignment): {test_acc_adapted:.4f}")
        print(f"{'='*70}")
        
        print("\nTest Confusion Matrix:")
        print(test_conf_matrix)
        print("\nTest Classification Report:")
        print(classification_report(test_true_labels, test_predictions))
        
        # ===== Save results =====
        result_dir = f'{self.testrun_foldername}result_subject_{self.args.leftout_subject}_adapt/'
        os.makedirs(result_dir, exist_ok=True)
        
        adapt_pkl = {
            'adapt_ratio': self.args.adapt_ratio,
            'support_size': support_size,
            'val_support_indices': val_support_indices,
            'test_support_indices': test_support_indices,
            'test_remaining_indices': test_indices_remain,
            'mean_train': mean_train,
            'var_train': var_train,
            'mean_0': mean_0,
            'var_0': var_0,
            'predictions': test_predictions,
            'labels': test_true_labels,
            'features_aligned': test_features_aligned,
            'accuracy': test_acc_adapted,
            'confusion_matrix': test_conf_matrix,
            'gesture_labels': self.gesture_labels,
        }
        
        adapt_pkl_file = f'{result_dir}adaptation_results.pkl'
        with open(adapt_pkl_file, 'wb') as f:
            pickle.dump(adapt_pkl, f)
        
        print(f"\n✓ Adaptation results saved to: {adapt_pkl_file}")
        
        # Save confusion matrix plot
        self.utils.plot_confusion_matrix(
            test_true_labels, test_predictions, self.gesture_labels,
            result_dir, self.args, 'test_adapted', 'test_adapted_confusion_matrix'
        )
        
        # Save summary
        summary_txt = f'{result_dir}adaptation_summary.txt'
        with open(summary_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"Few-shot + Transductive Adaptation Results\n")
            f.write(f"Subject {self.args.leftout_subject}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Adaptation Ratio: {self.args.adapt_ratio}\n")
            f.write(f"Support Set Size: {support_size * 2} (val: {support_size}, test: {support_size})\n")
            f.write(f"Test Samples Evaluated: {len(test_true_labels)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("Statistics\n")
            f.write("="*70 + "\n")
            f.write(f"Training Mean norm: {np.linalg.norm(mean_train):.4f}\n")
            f.write(f"Training Var mean: {var_train.mean():.6f}\n")
            f.write(f"Subject 0 Mean norm: {np.linalg.norm(mean_0):.4f}\n")
            f.write(f"Subject 0 Var mean: {var_0.mean():.6f}\n\n")
            
            f.write("="*70 + "\n")
            f.write("Metrics\n")
            f.write("="*70 + "\n")
            f.write(f"Test Accuracy (with alignment): {test_acc_adapted:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(test_conf_matrix) + "\n\n")
            f.write(classification_report(test_true_labels, test_predictions) + "\n")
        
        print(f"✓ Adaptation summary saved to: {summary_txt}")
        
        print(f"\n✓ All adaptation results saved to: {result_dir}\n")
        
        torch.cuda.empty_cache()
        
        return adapt_pkl
