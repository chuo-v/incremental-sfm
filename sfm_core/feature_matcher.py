from abc import ABC, abstractmethod
import numpy as np

from config import config


class BaseFeatureMatcher(ABC):
    """
    Abstract Base Class for all feature matching implementations.

    This establishes a strict data contract for the geometry engine:
    No matter what algorithm extracts or matches the features (SIFT, SuperPoint, etc.),
    it must output matched points as two aligned (N, 2) NumPy arrays.
    """

    def __init__(self):
        self.feature_cache = {}

    def _get_features(self, img: np.ndarray, idx: int | None):
        """
        Retrieves features from the cache if available, otherwise extracts and caches them.
        Enforces a maximum cache limit using LRU eviction to prevent OOM errors.

        Args:
            img (np.ndarray): The image array to extract features from if not cached.
            idx (int | None): The sequential index of the image, used as the cache key.
                              If None, caching is bypassed.

        Returns:
            tuple: The extracted keypoints and descriptors/tensors for the image.
        """
        if idx is not None and idx in self.feature_cache:
            # LRU Refresh
            features = self.feature_cache.pop(idx)
            self.feature_cache[idx] = features
            return features

        features = self._extract(img)

        if idx is not None:
            if len(self.feature_cache) >= config.max_cache_size:
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
            self.feature_cache[idx] = features

        return features

    @abstractmethod
    def _extract(self, img: np.ndarray):
        """
        Executes the algorithm-specific extraction on a single image.
        """
        pass

    def _bidirectional_match(
        self,
        matcher,
        kp1: tuple,
        kp2: tuple,
        des1: np.ndarray,
        des2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes bidirectional KNN matching, Lowe's Ratio Test, and Symmetric Cross-Check.
        """
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return np.empty((0, 2)), np.empty((0, 2))

        # Execute the forward KNN match
        knn_1_to_2 = matcher.knnMatch(des1, des2, k=2)

        # Lowe's Ratio Test (Forward)
        good_1_to_2 = {}
        for match_set in knn_1_to_2:
            if len(match_set) == 2:
                m, n = match_set
                if m.distance < config.lowe_ratio_threshold * n.distance:
                    good_1_to_2[m.queryIdx] = m.trainIdx

        if getattr(config, "disable_symmetric_matching", False):
            pts1, pts2 = [], []
            for idx1, idx2 in good_1_to_2.items():
                pts1.append(kp1[idx1].pt)
                pts2.append(kp2[idx2].pt)
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            if len(pts1) == 0:
                return np.empty((0, 2)), np.empty((0, 2))
            return pts1, pts2

        # Execute the backward KNN match if symmetric matching is enabled
        knn_2_to_1 = matcher.knnMatch(des2, des1, k=2)

        # Lowe's Ratio Test (Backward)
        good_2_to_1 = {}
        for match_set in knn_2_to_1:
            if len(match_set) == 2:
                m, n = match_set
                if m.distance < config.lowe_ratio_threshold * n.distance:
                    good_2_to_1[m.queryIdx] = m.trainIdx

        # Symmetric Cross-Check Intersection
        pts1, pts2 = [], []
        for idx1, idx2 in good_1_to_2.items():
            if good_2_to_1.get(idx2) == idx1:
                pts1.append(kp1[idx1].pt)
                pts2.append(kp2[idx2].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        if len(pts1) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        return pts1, pts2

    @abstractmethod
    def match(
        self, img1: np.ndarray, img2: np.ndarray, idx1: int = None, idx2: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts features from two images and finds reliable matches between them.
        Accepts optional image indices to pull from the feature cache.

        Args:
            img1 (np.ndarray): The first image (typically grayscale).
            img2 (np.ndarray): The second image (typically grayscale).
            idx1 (int, optional): The sequential index of the first image. Used to retrieve or store extracted features in the cache.
            idx2 (int, optional): The sequential index of the second image. Used to retrieve or store extracted features in the cache.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - pts1: An (N, 2) array of (x, y) coordinates in img1.
                - pts2: An (N, 2) array of corresponding (x, y) coordinates in img2.
                Note: The arrays must be aligned. pts1[i] must be the match for pts2[i].
        """
