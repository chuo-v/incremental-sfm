from pathlib import Path

import cv2
import numpy as np

from config import config


class SfMDataLoader:
    """
    Loads images sequentially and manages the camera intrinsic matrix (K).
    Automatically handles resolution downscaling and the corresponding
    mathematical adjustments to the focal length and optical center.
    """

    def __init__(self, dataset_name: str):
        """
        Initializes the data loader for a specific dataset.

        Args:
            dataset_name: The name of the dataset folder residing within DATA_DIR.
        """
        self.dataset_path = config.data_dir / dataset_name
        self.images_dir = self.dataset_path / "images"
        self.k_file = self.dataset_path / "K.txt"

        self._verify_paths()

        # Load and sort image paths sequentially
        self.image_paths = self._load_image_paths()

        # Load the base intrinsic matrix
        raw_K = self._load_intrinsics()

        # Scale the intrinsic matrix if downscaling is configured
        self.downscale_factor = config.downscale_factor
        if self.downscale_factor != 1.0:
            self.K = self._scale_intrinsics(raw_K, self.downscale_factor)
        else:
            self.K = raw_K

    def _verify_paths(self):
        """
        Ensures the expected directories and structural files exist.

        Raises:
            FileNotFoundError: If the dataset directory, images directory, or K.txt is missing.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.k_file.exists():
            raise FileNotFoundError(
                f"Intrinsic matrix file (K.txt) not found at: {self.k_file}"
            )

    def _load_image_paths(self) -> list[Path]:
        """
        Finds all standard image formats and sorts them alphabetically/sequentially.
        Assumes zero-padded filenames for correct sequential ordering.

        Returns:
            A sorted list of Path objects pointing to the dataset's images.

        Raises:
            ValueError: If no images are found in the target directory.
        """
        extensions = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for ext in extensions:
            paths.extend(self.images_dir.glob(ext))
            paths.extend(self.images_dir.glob(ext.upper()))  # Catch .JPG or .PNG

        if not paths:
            raise ValueError(f"No images found in {self.images_dir}")

        # Sorting guarantees Image N is matched against N+1
        return sorted(list(set(paths)))

    def _load_intrinsics(self) -> np.ndarray:
        """
        Reads the camera intrinsic matrix from a text file into a NumPy array.

        Returns:
            A 3x3 NumPy array representing the base intrinsic matrix.

        Raises:
            ValueError: If the file cannot be parsed as a space-separated numeric matrix.
        """
        try:
            return np.loadtxt(self.k_file)
        except (OSError, ValueError) as e:
            raise ValueError(
                f"Failed to parse K.txt. Ensure it is a 3x3 space-separated matrix. Error: {e}"
            )

    def _scale_intrinsics(self, K: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjusts the camera intrinsic matrix for downscaled images by scaling
        the focal lengths (fx, fy) and optical centers (cx, cy).

        Args:
            K: The original 3x3 intrinsic matrix.
            factor: The scaling multiplier (e.g., 0.5 to halve the resolution).

        Returns:
            K_scaled: The adjusted 3x3 intrinsic matrix.
        """
        K_scaled = K.copy()
        K_scaled[0, 0] *= factor  # fx
        K_scaled[1, 1] *= factor  # fy
        K_scaled[0, 2] *= factor  # cx
        K_scaled[1, 2] *= factor  # cy
        return K_scaled

    def __len__(self) -> int:
        """
        Returns the total number of images found in the dataset.
        """
        return len(self.image_paths)

    def get_image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the image at the given index, optionally resizes it based on the config,
        and converts it into both RGB and grayscale formats.

        Args:
            index: The sequential index of the image to retrieve.

        Returns:
            rgb_img: (H, W, 3) image array for color extraction and point cloud visualization.
            gray_img: (H, W) grayscale image array strictly for feature matching.

        Raises:
            ValueError: If the image cannot be read from disk.
        """
        img_path = str(self.image_paths[index])

        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            raise ValueError(f"Failed to load image at {img_path}")

        # Resize if necessary
        if self.downscale_factor != 1.0:
            new_width = int(bgr_img.shape[1] * self.downscale_factor)
            new_height = int(bgr_img.shape[0] * self.downscale_factor)
            bgr_img = cv2.resize(
                bgr_img, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        # Generate the standard formats needed by the pipeline
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        return rgb_img, gray_img
