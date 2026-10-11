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
        super().__init__()
        self.orb = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if getattr(config, "apply_clahe", False):
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self.clahe = None

    def _extract(self, img: np.ndarray):
        """
        Extracts ORB keypoints and descriptors from a single image,
        applying CLAHE if enabled in the config.
        """
        if self.clahe is not None:
            img = self.clahe.apply(img)
        return self.orb.detectAndCompute(img, None)

    def match(
        self, img1: np.ndarray, img2: np.ndarray, idx1: int = None, idx2: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves cached ORB features or extracts new ones from two grayscale images,
        then delegates matching to the bidirectional Brute-Force tracking pipeline.

        Args:
            img1: (H, W) grayscale array of the first image.
            img2: (H, W) grayscale array of the second image.
            idx1 (int, optional): The sequential index of the first image for cache retrieval/storage.
            idx2 (int, optional): The sequential index of the second image for cache retrieval/storage.

        Returns:
            pts1: (N, 2) array of matched (x, y) coordinates in the first image.
            pts2: (N, 2) array of matched (x, y) coordinates in the second image.
        """
        kp1, des1 = self._get_features(img1, idx1)
        kp2, des2 = self._get_features(img2, idx2)

        return self._bidirectional_match(self.matcher, kp1, kp2, des1, des2)
