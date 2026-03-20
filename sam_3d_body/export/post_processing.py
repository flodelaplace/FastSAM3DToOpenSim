"""
Post-processing utilities for keypoint sequences.
Ported from https://github.com/AitorIriondo/SAM3D-OpenSim
"""
import numpy as np


class PostProcessor:
    STANDARD_PROPORTIONS = {
        "upper_leg": 0.245,
        "lower_leg": 0.246,
        "upper_arm": 0.186,
        "forearm":   0.146,
    }

    def __init__(
        self,
        smooth_filter: bool = True,
        filter_cutoff: float = 6.0,
        filter_order: int = 4,
        normalize_bones: bool = True,
    ):
        self.smooth_filter = smooth_filter
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.normalize_bones = normalize_bones

    def process(self, keypoints: np.ndarray, fps: float = 30.0, subject_height: float = 1.75) -> np.ndarray:
        processed = keypoints.copy().astype(np.float64)
        processed = self._interpolate_missing(processed)
        if self.normalize_bones:
            processed = self._normalize_bones(processed, subject_height)
        if self.smooth_filter:
            processed = self._apply_butterworth(processed, fps)
        return processed

    def _interpolate_missing(self, keypoints):
        result = keypoints.copy()
        T, K, _ = result.shape
        for k in range(K):
            for dim in range(3):
                values = result[:, k, dim]
                valid_mask = ~np.isnan(values) & (values != 0)
                if np.sum(valid_mask) < 2:
                    continue
                valid_indices = np.where(valid_mask)[0]
                invalid_indices = np.where(~valid_mask)[0]
                if len(invalid_indices) > 0:
                    result[invalid_indices, k, dim] = np.interp(
                        invalid_indices, valid_indices, values[valid_indices]
                    )
        return result

    def _normalize_bones(self, keypoints, subject_height):
        result = keypoints.copy()
        expected_lengths = {
            (9,  11): subject_height * self.STANDARD_PROPORTIONS["upper_leg"],
            (11, 13): subject_height * self.STANDARD_PROPORTIONS["lower_leg"],
            (10, 12): subject_height * self.STANDARD_PROPORTIONS["upper_leg"],
            (12, 14): subject_height * self.STANDARD_PROPORTIONS["lower_leg"],
            (5,   7): subject_height * self.STANDARD_PROPORTIONS["upper_arm"],
            (7,  62): subject_height * self.STANDARD_PROPORTIONS["forearm"],
            (6,   8): subject_height * self.STANDARD_PROPORTIONS["upper_arm"],
            (8,  41): subject_height * self.STANDARD_PROPORTIONS["forearm"],
        }
        for t in range(result.shape[0]):
            for (parent_idx, child_idx), expected_len in expected_lengths.items():
                parent = result[t, parent_idx]
                child  = result[t, child_idx]
                vec = child - parent
                length = np.linalg.norm(vec)
                if length > 0.01:
                    scale = expected_len / length
                    if abs(scale - 1.0) > 0.2:
                        scale = np.clip(scale, 0.8, 1.2)
                        result[t, child_idx] = parent + vec * scale
        return result

    def _apply_butterworth(self, keypoints, fps):
        from scipy.signal import butter, filtfilt
        nyquist = fps / 2
        if self.filter_cutoff >= nyquist:
            return keypoints
        normalized_cutoff = self.filter_cutoff / nyquist
        b, a = butter(self.filter_order, normalized_cutoff, btype="low")
        result = keypoints.copy()
        T = result.shape[0]
        if T < 3 * self.filter_order + 1:
            return keypoints
        for k in range(result.shape[1]):
            for dim in range(3):
                try:
                    result[:, k, dim] = filtfilt(b, a, result[:, k, dim])
                except Exception:
                    pass
        return result

    def fix_left_right_swaps(self, keypoints):
        result = keypoints.copy()
        lr_pairs = [(9,10),(11,12),(13,14),(5,6),(7,8),(62,41)]
        for t in range(1, result.shape[0]):
            swap_detected = False
            for left_idx, right_idx in lr_pairs:
                prev_l, prev_r = result[t-1, left_idx], result[t-1, right_idx]
                curr_l, curr_r = result[t, left_idx],   result[t, right_idx]
                vel_normal  = np.linalg.norm(curr_l - prev_l) + np.linalg.norm(curr_r - prev_r)
                vel_swapped = np.linalg.norm(curr_r - prev_l) + np.linalg.norm(curr_l - prev_r)
                if vel_swapped < vel_normal * 0.5:
                    swap_detected = True
                    break
            if swap_detected:
                for left_idx, right_idx in lr_pairs:
                    result[t, [left_idx, right_idx]] = result[t, [right_idx, left_idx]]
        return result
