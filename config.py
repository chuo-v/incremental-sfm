import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Force load the .env file from the same directory as config.py
load_dotenv()


@dataclass
class PipelineConfig:
    """
    Central configuration object for the Incremental Structure from Motion (SfM) pipeline.
    Contains all tunable hyperparameters for image processing, feature extraction,
    matching thresholds, and geometric RANSAC tolerances.
    """

    # ------------------------------
    # Core Paths
    # ------------------------------
    data_dir: Path

    # ------------------------------
    # Image Processing
    # ------------------------------
    # Multiplier to resize input images to reduce processing time (e.g., 0.5 halves width and height).
    # The camera intrinsic matrix (K) is automatically scaled to match this factor.
    downscale_factor: float = 1.0

    # Maximum number of uncompressed image arrays to hold in RAM.
    max_cache_size: int = 15

    # ------------------------------
    # Feature Matching
    # ------------------------------
    # Maximum number of keypoints extracted per image (applies to SIFT, SuperPoint, and ORB).
    max_features: int = 8192

    # Threshold for Lowe's Ratio Test. Lower values strictly filter ambiguous matches.
    lowe_ratio_threshold: float = 0.75

    # Applies Contrast Limited Adaptive Histogram Equalization to enhance local contrast prior to extraction.
    apply_clahe: bool = True

    # Disables symmetric cross-check matching, halving feature matching time at the cost of outlier rejection.
    disable_symmetric_matching: bool = False

    # ------------------------------
    # Geometry & RANSAC Thresholds
    # ------------------------------
    # The confidence level (probability) that RANSAC successfully finds an outlier-free geometric model.
    ransac_prob: float = 0.999

    # Maximum allowable reprojection error (in pixels) for RANSAC inlier classification
    # and triangulated 3D point validation. Reduce accordingly if images are downscaled.
    ransac_threshold: float = 3.0

    # Minimum intersection angle (in degrees) of camera rays required to validate a triangulated 3D point.
    min_parallax_degrees: float = 1.5


def _load_config() -> PipelineConfig:
    """
    Parses system environment variables to initialize the global pipeline configuration.
    Requires the 'DATA_DIR' environment variable to be set, typically via a local .env file.

    Returns:
        PipelineConfig: An initialized configuration object containing the resolved dataset path.

    Raises:
        ValueError: If the 'DATA_DIR' environment variable is missing or unconfigured.
        FileNotFoundError: If the resolved 'DATA_DIR' path does not exist on the filesystem.
    """
    raw_data_dir = os.getenv("DATA_DIR")

    if not raw_data_dir:
        raise ValueError("DATA_DIR environment variable is missing or unconfigured.")

    data_dir_path = Path(raw_data_dir).resolve()

    if not data_dir_path.exists():
        raise FileNotFoundError(
            f"The configured DATA_DIR does not exist: {data_dir_path}"
        )

    return PipelineConfig(data_dir=data_dir_path)


# Instantiate a global config object to be imported across the repository
config = _load_config()
