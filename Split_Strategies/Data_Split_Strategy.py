"""
Data_Split_Strategy.py
- This file contains the base class for all data split strategies.
- Acts as a wrapper for X, Y, and label objects. Repeats function calls for each object.
"""
from .cross_validation_utilities import train_test_split as tts # custom train test split to split stratified without shuffling
import numpy as np
import torch 
class Data_Split_Strategy():
    """
    Base strategy. Serves as a wrapper to hold X, Y, and label objects. 
    """

    def __init__(self, X_data, Y_data, label_data, env):

        self.args =  env.args
        self.utils = env.utils
        self.leaveOut = env.leaveOut
        
        self.X = X_data
        self.Y = Y_data
        self.label = label_data

    def split(self):
        raise NotImplementedError("Subclasses must implement split()")
    
    # Helper Functions for Splits that are used in X, Y, and Label

    def convert_to_16_tensors(self, set_to_convert):
        self.X.convert_to_16_tensors(set_to_convert)
        self.Y.convert_to_16_tensors(set_to_convert)
        self.label.convert_to_16_tensors(set_to_convert)

    def concatenate_sessions(self, set_to_assign, set_to_concat):
        """
        Useful for when set_to_assign and set_to_concat are both instance variables. 
        """
        self.X.concatenate_sessions(set_to_assign, set_to_concat)
        self.Y.concatenate_sessions(set_to_assign, set_to_concat)
        self.label.concatenate_sessions(set_to_assign, set_to_concat)

    def del_data(self):
        self.X.del_data()
        self.Y.del_data()
        self.label.del_data()

    def print_set_shapes(self):
        self.X.print_set_shapes()
        self.Y.print_set_shapes()
        self.label.print_set_shapes()

    def validation_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.validation_from(X_new_data)
        self.Y.validation_from(Y_new_data)
        self.label.validation_from(label_new_data)

    def train_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.train_from(X_new_data)
        self.Y.train_from(Y_new_data)
        self.label.train_from(label_new_data)

    def train_from_self_tensor(self):
        self.X.set_to_self_tensor("train")
        self.Y.set_to_self_tensor("train")
        self.label.set_to_self_tensor("train")

    def validation_from_self_tensor(self):
        self.X.set_to_self_tensor("validation")
        self.Y.set_to_self_tensor("validation")
        self.label.set_to_self_tensor("validation")

    def test_from_validation(self):
        
        # TRAIN 
        self.X.test, self.X.validation, \
        self.Y.test, self.Y.validation, \
        self.label.test, self.label.validation \
        = tts.train_test_split(
            self.X.validation, 
            self.Y.validation, 
            test_size=0.5, 
            stratify=self.label.validation,
            random_state=self.args.seed,
            shuffle=(not self.args.train_test_split_for_time_series),
            force_regression=self.args.force_regression,
            transition_classifier=self.args.transition_classifier
        )

    def split_test_into_support_and_align(self, support_proportion=0.2):
        """
        Split the current test set into support and query according to support_proportion,
        then align the query samples to the support statistics.

        Formula implemented per request:
            Xnormalized = (Xquery - mu_target) / sigma_target
            Xaligned = Xnormalized * sigma_support + mu_support

        Uses the pre-split validation+test (target domain) statistics as mu_target/sigma_target
        (i.e., concatenation of current self.test and self.validation) and computes support
        statistics from the sampled support subset.
        """

        # If there's no test data, nothing to do
        if not hasattr(self.X, 'test') or self.X.test is None:
            return

        # Determine proportion for support; support_proportion is fraction of test to keep as support
        prop = support_proportion

        # Use custom train_test_split to split self.X.test into support and query
        X_support, X_query, Y_support, Y_query, label_support, label_query = tts.train_test_split(
            self.X.test,
            self.Y.test,
            test_size=1.0 - prop,
            stratify=self.label.test,
            random_state=self.args.seed,
            shuffle=(not self.args.train_test_split_for_time_series),
            force_regression=self.args.force_regression,
            transition_classifier=self.args.transition_classifier
        )

        # Compute target statistics from the training set (source of target domain statistics)
        # If training set is a torch tensor, convert to numpy
        if hasattr(self.X, 'train') and self.X.train is not None:
            if isinstance(self.X.train, torch.Tensor):
                target_all = self.X.train.cpu().detach().numpy()
            else:
                target_all = np.array(self.X.train)
        else:
            # fallback: use test + validation if train not available
            try:
                target_all = np.concatenate((self.X.test, self.X.validation), axis=0)
            except Exception:
                target_all = np.array(self.X.test)

        # Compute per-pixel/channel mean and std across samples
        mu_target = np.mean(target_all, axis=0)
        sigma_target = np.std(target_all, axis=0)

        # Compute support stats
        mu_support = np.mean(np.array(X_support), axis=0)
        sigma_support = np.std(np.array(X_support), axis=0)

        # Avoid divide by zero
        eps = 1e-6
        sigma_target = np.where(sigma_target == 0, eps, sigma_target)
        sigma_support = np.where(sigma_support == 0, eps, sigma_support)

        # Align query samples
        X_query = np.array(X_query)
        X_normalized = (X_query - mu_target) / sigma_target
        X_aligned = X_normalized * sigma_support + mu_support

        # Assign back: set test to aligned queries and keep support if needed
        # Preserve original types (numpy or torch) and device/dtype for tensors
        original_X_test = getattr(self.X, 'test', None)

        # helper to convert numpy array back to original type
        def _to_original_type(original, arr_np):
            if original is None:
                return arr_np
            # If original is a torch tensor, convert numpy -> torch with same dtype/device
            if isinstance(original, torch.Tensor):
                torch_to_np = {
                    torch.float16: np.float16,
                    torch.float32: np.float32,
                    torch.float64: np.float64,
                    torch.int64: np.int64,
                    torch.int32: np.int32,
                    torch.uint8: np.uint8,
                    torch.int16: np.int16,
                }
                npdtype = torch_to_np.get(original.dtype, np.float32)
                return torch.from_numpy(arr_np.astype(npdtype)).to(device=original.device, dtype=original.dtype)
            # If original has a dtype attribute (e.g., numpy array), try astype to match
            if hasattr(original, 'dtype'):
                try:
                    return arr_np.astype(original.dtype)
                except Exception:
                    return arr_np
            return arr_np

        self.X.test_support = _to_original_type(original_X_test, np.array(X_support))
        self.X.test_query = _to_original_type(original_X_test, X_query)
        self.X.test = _to_original_type(original_X_test, X_aligned)

        # Also store labels for support/query (keep them as numpy or convert to torch if original was)
        original_Y_test = getattr(self.Y, 'test', None)
        original_label_test = getattr(self.label, 'test', None)

        self.Y.test_support = _to_original_type(original_Y_test, np.array(Y_support))
        self.Y.test_query = _to_original_type(original_Y_test, np.array(Y_query))
        self.Y.test = _to_original_type(original_Y_test, np.array(self.Y.test))  # keep as before

        self.label.test_support = _to_original_type(original_label_test, np.array(label_support))
        self.label.test_query = _to_original_type(original_label_test, np.array(label_query))
        self.label.test = _to_original_type(original_label_test, np.array(self.label.test))

    def all_sets_to_tensor(self):
        self.X.all_sets_to_tensor()
        self.Y.all_sets_to_tensor()
        self.label.all_sets_to_tensor()

    def contract_to_binary_gestures(self):
        '''
        Convert from labels of the type [start, end] to [0] or [1] depending on whether or not they are a gesture. 
        '''

        self.Y.transition_labels_to_binary()
        self.label.transition_labels_to_binary()
