import cv2
import numpy as np

from config import config
from sfm_core.feature_matcher import BaseFeatureMatcher


class SIFTMatcher(BaseFeatureMatcher):
    """
    Implements the BaseFeatureMatcher contract using OpenCV's SIFT and FLANN.
    Includes optional CLAHE pre-processing and utilizes symmetric bidirectional
    matching for strict outlier rejection and drift reduction.
    """

    def __init__(self):
        """
        Initializes the SIFT feature extractor and FLANN-based KD-Tree matcher.
        Optionally sets up the CLAHE pre-processor if enabled in the global configuration.
        """
        self.sift = cv2.SIFT_create(nfeatures=config.max_features)

        # FLANN parameters for SIFT (using KD-Tree algorithm)
        # algorithm=0 corresponds to FLANN_INDEX_KDTREE
        index_params = {"algorithm": 0, "trees": 5}
        search_params = {"checks": 50}
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Initialize CLAHE if enabled in config
        if getattr(config, "apply_clahe", False):
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self.clahe = None

    def _to_root_sift(self, des: np.ndarray) -> np.ndarray:
        """
        Converts standard SIFT descriptors to RootSIFT.
        """
        if des is None or len(des) == 0:
            return des

        # Apply L1 normalization (sum along the feature axis)
        l1_norm = np.linalg.norm(des, ord=1, axis=1, keepdims=True)
        # Prevent division by zero for any empty descriptors
        l1_norm[l1_norm == 0] = 1e-7
        # Element-wise square root on the normalized descriptors
        root_des = np.sqrt(des / l1_norm)

        return root_des.astype(np.float32)

    def match(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects and extracts SIFT features from two grayscale images, then delegates
        matching to the bidirectional FLANN tracking pipeline.

        Args:
            img1: (H, W) grayscale array of the first image.
            img2: (H, W) grayscale array of the second image.

        Returns:
            pts1: (N, 2) array of matched (x, y) coordinates in the first image.
            pts2: (N, 2) array of matched (x, y) coordinates in the second image.
        """
        # Pre-process with CLAHE to enhance local contrast for better gradients
        if self.clahe is not None:
            img1 = self.clahe.apply(img1)
            img2 = self.clahe.apply(img2)

        # Extract keypoints (kp) and descriptors (des)
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # Apply RootSIFT conversion if enabled
        if getattr(config, "use_root_sift", False):
            des1 = self._to_root_sift(des1)
            des2 = self._to_root_sift(des2)

        pts1, pts2 = self._bidirectional_match(self.matcher, kp1, kp2, des1, des2)

        return pts1, pts2
