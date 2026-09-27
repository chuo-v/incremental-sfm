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

    def match(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects and matches features between two images using SuperPoint and LightGlue.
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

        # Convert NumPy arrays to PyTorch tensors
        tensor1 = self._preprocess_image(img1)
        tensor2 = self._preprocess_image(img2)

        with torch.no_grad():
            # Extract keypoints and descriptors
            feats1 = self.extractor.extract(tensor1)
            feats2 = self.extractor.extract(tensor2)
            # Perform feature matching
            matches01 = self.matcher({"image0": feats1, "image1": feats2})

        # Remove the batch dimension using LightGlue's utility function
        feats1 = rbd(feats1)
        feats2 = rbd(feats2)
        matches01 = rbd(matches01)

        # Extract the matched index pairs (M, 2)
        match_indices = matches01["matches"]

        # Return empty arrays if no matches survive
        if len(match_indices) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        # Gather the matched keypoint coordinates and push back to CPU/NumPy
        pts1 = feats1["keypoints"][match_indices[..., 0]].cpu().numpy()
        pts2 = feats2["keypoints"][match_indices[..., 1]].cpu().numpy()

        return pts1, pts2
