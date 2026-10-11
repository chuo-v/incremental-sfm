import logging
import cv2
import torch
import numpy as np

from config import config
from sfm_core.feature_matcher import BaseFeatureMatcher
from lightglue import SuperPoint, LightGlue
from lightglue.utils import rbd

logger = logging.getLogger(__name__)


class SuperPointMatcher(BaseFeatureMatcher):
    """
    Implements the BaseFeatureMatcher contract using deep learning-based
    SuperPoint feature extraction and LightGlue matching.
    """

    def __init__(self):
        """
        Initializes the SuperPoint extractor and LightGlue matcher neural networks.
        Automatically detects and assigns the models to the optimal available
        hardware accelerator (CUDA, MPS, or CPU) and configures optional pre-processing.
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Initializing SuperPoint & LightGlue on device: {self.device}")

        # Initialize the feature extractor and matcher, explicitly setting to eval mode
        self.extractor = (
            SuperPoint(max_num_keypoints=config.max_features).eval().to(self.device)
        )
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        # Initialize CLAHE if enabled in config
        if getattr(config, "apply_clahe", False):
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self.clahe = None

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Converts an OpenCV grayscale (H, W) array into a normalized PyTorch tensor.
        Args:
            img: (H, W) 8-bit grayscale image array.
        Returns:
            torch.Tensor: Normalized float32 tensor of shape (1, 1, H, W) located on the target device.
        """
        tensor = torch.from_numpy(img).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def _extract(self, img: np.ndarray):
        """
        Extracts SuperPoint feature tensors from a single image.
        Pre-processes the image into a normalized PyTorch tensor before extraction.
        """
        if self.clahe is not None:
            img = self.clahe.apply(img)
        tensor = self._preprocess_image(img)
        with torch.no_grad():
            feats = self.extractor.extract(tensor)
        return feats

    def match(
        self, img1: np.ndarray, img2: np.ndarray, idx1: int = None, idx2: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves cached SuperPoint features or extracts new ones from two grayscale images,
        then delegates matching to the LightGlue neural network.

        Args:
            img1: (H, W) grayscale array of the first image.
            img2: (H, W) grayscale array of the second image.
            idx1 (int, optional): The sequential index of the first image for cache retrieval/storage.
            idx2 (int, optional): The sequential index of the second image for cache retrieval/storage.

        Returns:
            pts1: (N, 2) array of matched (x, y) coordinates in the first image.
            pts2: (N, 2) array of matched (x, y) coordinates in the second image.
        """
        feats1 = self._get_features(img1, idx1)
        feats2 = self._get_features(img2, idx2)

        with torch.no_grad():
            # Perform feature matching
            matches01 = self.matcher({"image0": feats1, "image1": feats2})

        # Remove the batch dimension using LightGlue's utility function
        f1 = rbd(feats1)
        f2 = rbd(feats2)
        matches01 = rbd(matches01)

        # Extract the matched index pairs (M, 2)
        match_indices = matches01["matches"]

        # Return empty arrays if no matches survive
        if len(match_indices) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        # Gather the matched keypoint coordinates and push back to CPU/NumPy
        pts1 = f1["keypoints"][match_indices[..., 0]].cpu().numpy()
        pts2 = f2["keypoints"][match_indices[..., 1]].cpu().numpy()

        return pts1, pts2
