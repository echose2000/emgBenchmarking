from .Data_Split_Strategy import Data_Split_Strategy
import numpy as np
from .cross_validation_utilities import train_test_split as tts # custom train test split to split stratified without shuffling
import torch
from sklearn import preprocessing, model_selection

class Leave_One_Subject_Out(Data_Split_Strategy):
        
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
        self.wrap_rng = np.random.default_rng(self.args.seed)

    def _duplicate_finetune_with_gaussian_noise(self, X_data, Y_data, label_data):
        """
        Duplicate finetune samples by adding Gaussian noise and concatenating to original data.
        Only applies when augmentation is enabled and data is non-empty.
        """
        if not getattr(self.args, "augment_finetune_with_gaussian_noise", False):
            return X_data, Y_data, label_data

        if X_data is None or len(X_data) == 0:
            return X_data, Y_data, label_data

        noise_std = float(getattr(self.args, "finetune_gaussian_noise_std", 0.01))
        if noise_std <= 0:
            return X_data, Y_data, label_data

        if isinstance(X_data, torch.Tensor):
            X_float = X_data.to(torch.float32)
            noise = torch.randn_like(X_float) * noise_std
            X_noisy = (X_float + noise).to(X_data.dtype)
            X_aug = torch.cat((X_data, X_noisy), dim=0)

            Y_aug = torch.cat((Y_data, Y_data), dim=0)
            label_aug = torch.cat((label_data, label_data), dim=0)
        else:
            X_np = np.asarray(X_data)
            noise = np.random.normal(loc=0.0, scale=noise_std, size=X_np.shape).astype(np.float32)
            X_noisy = (X_np.astype(np.float32) + noise).astype(X_np.dtype, copy=False)
            X_aug = np.concatenate((X_np, X_noisy), axis=0)

            Y_np = np.asarray(Y_data)
            label_np = np.asarray(label_data)
            Y_aug = np.concatenate((Y_np, Y_np), axis=0)
            label_aug = np.concatenate((label_np, label_np), axis=0)

        print(f"[finetune-augment] Gaussian noise applied (std={noise_std}). Finetune samples: {len(X_data)} -> {len(X_aug)}")
        return X_aug, Y_aug, label_aug

    def _duplicate_finetune_with_channel_shift_25(self, X_data, Y_data, label_data):
        """
        Duplicate finetune samples by cyclically shifting the channel axis by 25%.
        For input shaped (N, 3, 128, 224), this shifts along axis=2 by 32.
        """
        if not getattr(self.args, "augment_finetune_with_channel_shift_25", False):
            return X_data, Y_data, label_data

        if X_data is None or len(X_data) == 0:
            return X_data, Y_data, label_data

        if isinstance(X_data, torch.Tensor):
            channel_axis = X_data.ndim - 2
            channel_count = int(X_data.shape[channel_axis])
            shift = max(1, int(round(channel_count * 0.25)))
            X_shifted = torch.roll(X_data, shifts=shift, dims=channel_axis)
            X_aug = torch.cat((X_data, X_shifted), dim=0)
            Y_aug = torch.cat((Y_data, Y_data), dim=0)
            label_aug = torch.cat((label_data, label_data), dim=0)
        else:
            X_np = np.asarray(X_data)
            channel_axis = X_np.ndim - 2
            channel_count = int(X_np.shape[channel_axis])
            shift = max(1, int(round(channel_count * 0.25)))
            X_shifted = np.roll(X_np, shift=shift, axis=channel_axis)
            X_aug = np.concatenate((X_np, X_shifted), axis=0)

            Y_np = np.asarray(Y_data)
            label_np = np.asarray(label_data)
            Y_aug = np.concatenate((Y_np, Y_np), axis=0)
            label_aug = np.concatenate((label_np, label_np), axis=0)

        print(
            f"[finetune-augment] Channel shift 25% applied "
            f"(axis={channel_axis}, channels={channel_count}, shift={shift}). "
            f"Finetune samples: {len(X_data)} -> {len(X_aug)}"
        )
        return X_aug, Y_aug, label_aug

    def _duplicate_finetune_with_magnitude_warp(self, X_data, Y_data, label_data):
        """
        Duplicate finetune samples with smooth magnitude warping on the last axis.
        For input shaped (N, 3, 128, 224), this applies a smooth length-224 curve
        (mean normalized to 1) to each sample and broadcasts over channels.
        """
        if not getattr(self.args, "augment_finetune_with_magnitude_warp", False):
            return X_data, Y_data, label_data

        if X_data is None or len(X_data) == 0:
            return X_data, Y_data, label_data

        # Keep curve variance controlled: small random fluctuation, clipped range, then mean=1 normalization.
        knot_count = 6
        warp_std = 0.12
        warp_min, warp_max = 0.8, 1.3
        smooth_kernel = 9

        if isinstance(X_data, torch.Tensor):
            X_ref = X_data.detach().cpu().numpy()
            is_torch = True
        else:
            X_ref = np.asarray(X_data)
            is_torch = False

        time_axis_len = int(X_ref.shape[-1])
        sample_count = int(X_ref.shape[0])

        knot_x = np.linspace(0, time_axis_len - 1, num=knot_count)
        target_x = np.arange(time_axis_len)
        knot_values = np.random.normal(loc=1.0, scale=warp_std, size=(sample_count, knot_count))
        knot_values = np.clip(knot_values, warp_min, warp_max)

        curves = np.stack([np.interp(target_x, knot_x, knot_values[i]) for i in range(sample_count)], axis=0)

        if smooth_kernel > 1:
            kernel = np.ones(smooth_kernel, dtype=np.float32) / float(smooth_kernel)
            curves = np.stack([
                np.convolve(curves[i], kernel, mode='same') for i in range(sample_count)
            ], axis=0)

        curves = curves / (np.mean(curves, axis=1, keepdims=True) + 1e-6)
        curves = np.clip(curves, warp_min, warp_max).astype(np.float32)

        broadcast_shape = (sample_count,) + (1,) * (X_ref.ndim - 2) + (time_axis_len,)
        curves = curves.reshape(broadcast_shape)

        X_warped_np = (X_ref.astype(np.float32) * curves).astype(X_ref.dtype, copy=False)

        if is_torch:
            X_warped = torch.from_numpy(X_warped_np).to(X_data.dtype)
            X_aug = torch.cat((X_data, X_warped), dim=0)
            Y_aug = torch.cat((Y_data, Y_data), dim=0)
            label_aug = torch.cat((label_data, label_data), dim=0)
        else:
            X_aug = np.concatenate((X_ref, X_warped_np), axis=0)
            Y_np = np.asarray(Y_data)
            label_np = np.asarray(label_data)
            Y_aug = np.concatenate((Y_np, Y_np), axis=0)
            label_aug = np.concatenate((label_np, label_np), axis=0)

        print(
            f"[finetune-augment] Magnitude warp applied "
            f"(time_axis={time_axis_len}, curve_mean~1.0, range=[{warp_min}, {warp_max}]). "
            f"Finetune samples: {len(X_data)} -> {len(X_aug)}"
        )
        return X_aug, Y_aug, label_aug

    def _duplicate_finetune_with_wrap(self, X_data, Y_data, label_data):
        """
        Duplicate finetune samples by seeded wrap augmentation.
        For each trial, sample a single factor in [0.9, 1.1], apply the same factor to
        all channels of that trial, then resize back to the original length on last axis.
        """
        if not getattr(self.args, "augment_finetune_with_wrap", False):
            return X_data, Y_data, label_data

        if X_data is None or len(X_data) == 0:
            return X_data, Y_data, label_data

        if isinstance(X_data, torch.Tensor):
            X_ref = X_data.detach().cpu().numpy()
            is_torch = True
        else:
            X_ref = np.asarray(X_data)
            is_torch = False

        X_float = X_ref.astype(np.float32)
        sample_count = int(X_float.shape[0])
        time_len = int(X_float.shape[-1])
        X_wrap = np.empty_like(X_float, dtype=np.float32)

        factors = []
        for i in range(sample_count):
            factor = float(self.wrap_rng.uniform(0.9, 1.1))
            factors.append(factor)

            warped_len = max(2, int(round(time_len * factor)))
            src = X_float[i].reshape(-1, time_len)

            # Step 1: warp with factor (change temporal density)
            x_warp = np.linspace(0, time_len - 1, num=warped_len, dtype=np.float32)
            x_orig = np.arange(time_len, dtype=np.float32)
            warped = np.stack([
                np.interp(x_warp, x_orig, channel) for channel in src
            ], axis=0)

            # Step 2: resize warped signal back to original length
            x_back = np.linspace(0, warped_len - 1, num=time_len, dtype=np.float32)
            x_warp_idx = np.arange(warped_len, dtype=np.float32)
            restored = np.stack([
                np.interp(x_back, x_warp_idx, channel) for channel in warped
            ], axis=0)

            X_wrap[i] = restored.reshape(X_float[i].shape)

        X_wrap = X_wrap.astype(X_ref.dtype, copy=False)

        if is_torch:
            X_wrap_t = torch.from_numpy(X_wrap).to(X_data.dtype)
            X_aug = torch.cat((X_data, X_wrap_t), dim=0)
            Y_aug = torch.cat((Y_data, Y_data), dim=0)
            label_aug = torch.cat((label_data, label_data), dim=0)
        else:
            X_aug = np.concatenate((X_ref, X_wrap), axis=0)
            Y_np = np.asarray(Y_data)
            label_np = np.asarray(label_data)
            Y_aug = np.concatenate((Y_np, Y_np), axis=0)
            label_aug = np.concatenate((label_np, label_np), axis=0)

        print(
            f"[finetune-augment] Wrap applied (factor in [0.9, 1.1], seeded). "
            f"factor_mean={np.mean(factors):.4f}. Finetune samples: {len(X_data)} -> {len(X_aug)}"
        )
        return X_aug, Y_aug, label_aug

    def split(self):

        self.train_from_non_left_out_subj()
        self.validation_from_leave_out_subj()
        self.convert_datasets()

        if self.args.transfer_learning:
            self.adjust_sets_for_leave_out_subj()

        super().test_from_validation()
        super().print_set_shapes()
        super().all_sets_to_tensor()

        if self.args.transition_classifier: 
            super().contract_to_binary_gestures()

    def append_to_train_unlabeled_list(self, X_new_data, Y_new_data, label_new_data):
        self.X.append_to_train_unlabeled_list(X_new_data)
        self.Y.append_to_train_unlabeled_list(Y_new_data)
        self.label.append_to_train_unlabeled_list(label_new_data)

    def append_to_train_list(self, X_new_data, Y_new_data, label_new_data):
        self.X.append_to_train_list(X_new_data)
        self.Y.append_to_train_list(Y_new_data)
        self.label.append_to_train_list(label_new_data)
    
    def convert_datasets(self):
        super().concatenate_sessions(set_to_assign="train", set_to_concat="train_list")
        super().convert_to_16_tensors(set_to_convert="train")
        
        if self.args.proportion_unlabeled_data_from_training_subjects>0:
            super().concatenate_sessions(set_to_assign="train_unlabeled", set_to_concat="train_unlabeled_list")
            super().convert_to_16_tensors(set_to_convert="train_unlabeled")

        super().convert_to_16_tensors(set_to_convert="validation")

    def concatenate_to_train(self, X_new_data, Y_new_data, label_new_data):
        self.X.concatenate_to_train(X_new_data)
        self.Y.concatenate_to_train(Y_new_data)
        self.label.concatenate_to_train(label_new_data) 

    def concatenate_to_train_unlabeled(self, X_new_data, Y_new_data, label_new_data):
        self.X.concatenate_to_train_unlabeled(X_new_data)
        self.Y.concatenate_to_train_unlabeled(Y_new_data)
        self.label.concatenate_to_train_unlabeled(label_new_data)

    def train_unlabeled_from_self_tensor(self):
        self.X.set_to_self_tensor("train_unlabeled")
        self.Y.set_to_self_tensor("train_unlabeled")
        self.label.set_to_self_tensor("train_unlabeled")
   

    def train_from_non_left_out_subj(self):

        total_windows = 0
        cumulative_sizes = []

        for i in range(len(self.X.data)):
            if i == self.leaveOut-1:
                continue

            X_train_temp = np.array(self.X.data[i])
            Y_train_temp = np.array(self.Y.data[i])
            label_train_temp = np.array(self.label.data[i])

            if self.args.reduce_training_data_size:
                
                reduced_size_per_subject = self.args.reduced_training_data_size // (self.utils.num_subjects - 1)
                proportion_to_keep = reduced_size_per_subject / X_train_temp.shape[0]
                X_train_temp, _, \
                Y_train_temp, _, \
                label_train_temp, _ \
                = model_selection.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=proportion_to_keep, 
                    stratify=label_train_temp, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series)
                )

            if self.args.proportion_data_from_training_subjects < 1.0:
                X_train_temp, _, \
                Y_train_temp, _, \
                label_train_temp, _ \
                = tts.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=self.args.proportion_data_from_training_subjects, 
                    stratify=label_train_temp, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series),
                    force_regression=self.args.force_regression, 
                    transition_classifier=self.args.transition_classifier
                )
 
            if self.args.proportion_unlabeled_data_from_training_subjects>0:
                X_train_labeled, X_train_unlabeled, \
                Y_train_labeled, Y_train_unlabeled, \
                label_train_labeled, label_train_unlabeled = tts.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=1-self.args.proportion_unlabeled_data_from_training_subjects, 
                    stratify=label_train_temp, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series), 
                    force_regression=self.args.force_regression, 
                    transition_classifier=self.args.transition_classifier
                )

                self.append_to_train_list(X_train_labeled, Y_train_labeled, label_train_labeled)

                self.append_to_train_unlabeled_list(X_train_unlabeled, Y_train_unlabeled, label_train_unlabeled)

            else:
                # TRAIN 
                # TODO: Should stratify if doing transition classifier
                if self.args.transition_classifier:
   
                    Y_train_temp = torch.from_numpy(Y_train_temp).to(torch.float16)
                    label_train_temp = torch.from_numpy(label_train_temp).to(torch.float16)

                    if self.args.transition_classifier: 
                        X_train_temp, _, \
                        Y_train_temp, _, \
                        label_train_temp, _ \
                        = tts.train_test_split(
                            X_train_temp, 
                            Y_train_temp, 
                            train_size=1.0, 
                            stratify=label_train_temp, 
                            random_state=self.args.seed, 
                            shuffle=(not self.args.train_test_split_for_time_series), 
                            force_regression=self.args.force_regression, 
                            transition_classifier=self.args.transition_classifier
                        )


                total_windows += X_train_temp.shape[0]

                self.append_to_train_list(X_train_temp, Y_train_temp, label_train_temp)

            cumulative_sizes.append(total_windows)

        self.X.cumulative_sizes = cumulative_sizes

    def validation_from_leave_out_subj(self):
        self.X.validation_from_leave_out_subj()
        self.Y.validation_from_leave_out_subj()
        self.label.validation_from_leave_out_subj()

    def train_finetuning_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.train_finetuning_from(X_new_data)
        self.Y.train_finetuning_from(Y_new_data)
        self.label.train_finetuning_from(label_new_data)

    def train_finetuning_unlabeled_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.train_finetuning_unlabeled_from(X_new_data)
        self.Y.train_finetuning_unlabeled_from(Y_new_data)
        self.label.train_finetuning_unlabeled_from(label_new_data)

    def adjust_sets_for_leave_out_subj(self): 
        """
        If doing transfer learning, splits left out subject's data into train_partial_leftout_subject and validation_partial_leftout_subject sets.
        """

        assert self.args.transfer_learning, "Transfer learning must be turned on to split left out subject's data."

        proportion_to_keep_of_leftout_subject_for_training = self.args.proportion_transfer_learning_from_leftout_subject
        
        proportion_unlabeled_of_proportion_to_keep_of_leftout = self.args.proportion_unlabeled_data_from_leftout_subject

        proportion_unlabeled_of_training_subjects = self.args.proportion_unlabeled_data_from_training_subjects
        
        # TRAIN AND VALIDATION 
        # Split leftout validation into train and validation
        if proportion_to_keep_of_leftout_subject_for_training>0.0:
            X_train_partial_leftout_subject, X_validation_partial_leftout_subject, \
            Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject, \
            label_train_partial_leftout_subject, label_validation_partial_leftout_subject = \
                tts.train_test_split(
                    self.X.validation, 
                    self.Y.validation, 
                    train_size=proportion_to_keep_of_leftout_subject_for_training, 
                    stratify=self.label.validation, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series), 
                    force_regression=self.args.force_regression, 
                    transition_classifier=self.args.transition_classifier   
                )

        # Otherwise validate with all of left out subject's data
        else:
            X_validation_partial_leftout_subject = self.X.validation
            Y_validation_partial_leftout_subject = self.Y.validation
            label_validation_partial_leftout_subject = self.label.validation

            X_train_partial_leftout_subject = torch.tensor([])
            Y_train_partial_leftout_subject = torch.tensor([])
            label_train_partial_leftout_subject = torch.tensor([])

        # If unlabeled domain adaptation, split the training data into labeled and unlabeled
        if self.args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep_of_leftout>0:

            X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
            Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject, \
            label_train_labeled_partial_leftout_subject, label_train_unlabeled_partial_leftout_subject = \
                tts.train_test_split(
                    X_train_partial_leftout_subject, 
                    Y_train_partial_leftout_subject, 
                    train_size=1-proportion_unlabeled_of_proportion_to_keep_of_leftout, 
                    stratify=label_train_partial_leftout_subject, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series), 
                    force_regression=self.args.force_regression, 
                    transition_classifier=self.args.transition_classifier
                )

        # Add the partial from leftout subject to train/finetune
        if not self.args.turn_on_unlabeled_domain_adaptation:
            # Append the partial validation data to the training data
            if proportion_to_keep_of_leftout_subject_for_training>0:
                
                if not self.args.pretrain_and_finetune:
                    self.concatenate_to_train(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)
            
                else:
                    X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_gaussian_noise(
                        X_train_partial_leftout_subject,
                        Y_train_partial_leftout_subject,
                        label_train_partial_leftout_subject
                    )
                    X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_channel_shift_25(
                        X_train_partial_leftout_subject,
                        Y_train_partial_leftout_subject,
                        label_train_partial_leftout_subject
                    )
                    X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_magnitude_warp(
                        X_train_partial_leftout_subject,
                        Y_train_partial_leftout_subject,
                        label_train_partial_leftout_subject
                    )
                    X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_wrap(
                        X_train_partial_leftout_subject,
                        Y_train_partial_leftout_subject,
                        label_train_partial_leftout_subject
                    )
                    self.train_finetuning_from(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)
         
        else: # unlabeled domain adaptation
            if proportion_unlabeled_of_training_subjects>0:
                self.train_from_self_tensor()
                self.train_unlabeled_from_self_tensor()


            if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                if proportion_unlabeled_of_proportion_to_keep_of_leftout==0:
                    X_train_labeled_partial_leftout_subject = X_train_partial_leftout_subject
                    Y_train_labeled_partial_leftout_subject = Y_train_partial_leftout_subject
                    label_train_labeled_partial_leftout_subject = label_train_partial_leftout_subject

                if self.args.pretrain_and_finetune:
                    X_train_labeled_partial_leftout_subject, Y_train_labeled_partial_leftout_subject, label_train_labeled_partial_leftout_subject = self._duplicate_finetune_with_gaussian_noise(
                        X_train_labeled_partial_leftout_subject,
                        Y_train_labeled_partial_leftout_subject,
                        label_train_labeled_partial_leftout_subject
                    )
                    X_train_labeled_partial_leftout_subject, Y_train_labeled_partial_leftout_subject, label_train_labeled_partial_leftout_subject = self._duplicate_finetune_with_channel_shift_25(
                        X_train_labeled_partial_leftout_subject,
                        Y_train_labeled_partial_leftout_subject,
                        label_train_labeled_partial_leftout_subject
                    )
                    X_train_labeled_partial_leftout_subject, Y_train_labeled_partial_leftout_subject, label_train_labeled_partial_leftout_subject = self._duplicate_finetune_with_magnitude_warp(
                        X_train_labeled_partial_leftout_subject,
                        Y_train_labeled_partial_leftout_subject,
                        label_train_labeled_partial_leftout_subject
                    )
                    X_train_labeled_partial_leftout_subject, Y_train_labeled_partial_leftout_subject, label_train_labeled_partial_leftout_subject = self._duplicate_finetune_with_wrap(
                        X_train_labeled_partial_leftout_subject,
                        Y_train_labeled_partial_leftout_subject,
                        label_train_labeled_partial_leftout_subject
                    )
                    self.train_finetuning_from(X_train_labeled_partial_leftout_subject, Y_train_labeled_partial_leftout_subject, label_train_labeled_partial_leftout_subject)

                    self.train_finetuning_unlabeled_from(X_train_unlabeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject, label_train_unlabeled_partial_leftout_subject)

                else: 
                    self.concatenate_to_train(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)

                    self.train_from_self_tensor()

                    self.concatenate_to_train_unlabeled(X_train_unlabeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject, label_train_unlabeled_partial_leftout_subject)
                    self.train_unlabeled_from_self_tensor()
                  
            else:
                if proportion_to_keep_of_leftout_subject_for_training>0:
                    if not self.args.pretrain_and_finetune:
                        self.concatenate_to_train(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)
                        self.train_from_self_tensor()
                    
                    else: 
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_gaussian_noise(
                            X_train_partial_leftout_subject,
                            Y_train_partial_leftout_subject,
                            label_train_partial_leftout_subject
                        )
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_channel_shift_25(
                            X_train_partial_leftout_subject,
                            Y_train_partial_leftout_subject,
                            label_train_partial_leftout_subject
                        )
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_magnitude_warp(
                            X_train_partial_leftout_subject,
                            Y_train_partial_leftout_subject,
                            label_train_partial_leftout_subject
                        )
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject = self._duplicate_finetune_with_wrap(
                            X_train_partial_leftout_subject,
                            Y_train_partial_leftout_subject,
                            label_train_partial_leftout_subject
                        )
                        self.train_finetuning_from(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)

        # Update the validation data
        self.train_from_self_tensor()
        self.validation_from(X_validation_partial_leftout_subject, Y_validation_partial_leftout_subject, label_validation_partial_leftout_subject)
        
        del X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject, label_train_partial_leftout_subject, label_validation_partial_leftout_subject
        
    