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
        super().__init__()
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

    def _extract(self, img: np.ndarray):
        """
        Extracts SIFT keypoints and descriptors from a single image,
        applying CLAHE and RootSIFT conversions if enabled in the config.
        """
        if self.clahe is not None:
            img = self.clahe.apply(img)
        kp, des = self.sift.detectAndCompute(img, None)

        # Apply RootSIFT conversion if enabled
        if getattr(config, "use_root_sift", False):
            des = self._to_root_sift(des)

        return kp, des

    def match(
        self, img1: np.ndarray, img2: np.ndarray, idx1: int = None, idx2: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves cached SIFT features or extracts new ones from two grayscale images,
        then delegates matching to the bidirectional FLANN tracking pipeline.

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
