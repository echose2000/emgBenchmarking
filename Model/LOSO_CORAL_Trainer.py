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
        
        # Loss weights
        self.lambda1 = 0.1  # CORAL loss weight
        self.lambda2 = 0.5  # Prototype loss weight
    
    def setup_model(self):
        """Set up all model parameters for the run."""
        super().set_pretrain_path()
        self.set_model()
        self.set_optimizer()
        self.set_param_requires_grad()
        super().set_resize_transform()
        self.set_loaders()  # Custom loaders for LOSO
        self.set_criterion()
        
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
