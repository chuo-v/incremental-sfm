import cv2
import numpy as np

from config import config
from sfm_core.feature_matcher import BaseFeatureMatcher


class ORBMatcher(BaseFeatureMatcher):
    """
    Implements the BaseFeatureMatcher contract using OpenCV's ORB (Oriented FAST
    and Rotated BRIEF) and Brute-Force Hamming matching. Includes optional CLAHE
    pre-processing and utilizes symmetric bidirectional matching for strict outlier
    rejection and drift reduction.
    """

    def __init__(self):
        """
        Initializes the ORB feature extractor and Brute-Force Hamming matcher.
        Optionally sets up the CLAHE pre-processor if enabled in the global configuration.
        """
        self.orb = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if getattr(config, "apply_clahe", False):
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self.clahe = None

    def match(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects and extracts ORB features from two grayscale images, then delegates
        matching to the bidirectional Brute-Force tracking pipeline.

        Args:
            img1: (H, W) grayscale array of the first image.
            img2: (H, W) grayscale array of the second image.

        Returns:
            pts1: (N, 2) array of matched (x, y) coordinates in the first image.
            pts2: (N, 2) array of matched (x, y) coordinates in the second image.
        """
        if self.clahe is not None:
            img1 = self.clahe.apply(img1)
            img2 = self.clahe.apply(img2)

        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        pts1, pts2 = self._bidirectional_match(self.matcher, kp1, kp2, des1, des2)

        return pts1, pts2
