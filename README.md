# incremental-sfm

`incremental-sfm` is an Incremental Structure from Motion (SfM) pipeline that orchestrates data loading, feature matching, geometric tracking, global relocalization, and 3D mapping. It processes sequential multi-view image datasets and exports colored 3D point clouds alongside run metrics.

## Features

* **Multiple Feature Matchers:** Supports OpenCV's SIFT and ORB, as well as deep learning-based SuperPoint paired with LightGlue.
* **Robust Tracking & Mapping:** Implements fast sequential N to N-1 tracking with automatic global relocalization using Perspective-n-Point (PnP) if tracking fails.
* **Dynamic Triangulation:** Utilizes both fast epipolar triangulation and multi-view Direct Linear Transform (DLT) with strict cheirality and parallax filtering.
* **Configurable Pipeline:** Exposes hyperparameters for image downscaling, Lowe's ratio test thresholds, optional CLAHE contrast pre-processing, and RANSAC tolerances via a central [`config.py`](./config.py) module.

## Installation

1. Ensure your environment runs Python 3.10 or higher.
2. Install the necessary dependencies (including OpenCV, NumPy, PyTorch, TorchVision, and LightGlue):
    ```bash
    pip install .
    ```
    *Note: For development tools like `pytest` and `ruff`, install with `pip install .[dev]`*.

3. Copy `.env.example` to `.env` and set the `DATA_DIR` variable to the directory where your datasets are stored.

## Dataset Structure
Each dataset must be placed inside your configured `DATA_DIR` and adhere to the following structure:

```
<DATA_DIR>/<dataset_folder_name>/
├── images/      # Contains the sequential image files (.jpg, .png)
└── K.txt        # A 3x3 space-separated camera intrinsic matrix
```

## Usage

Execute the pipeline via the command line using `main.py`.

```bash
python main.py --dataset <dataset_folder_name> --matcher <MATCHER_TYPE>
```
* **`--dataset`** *(Required)*: The exact name of the dataset folder residing inside your configured `DATA_DIR`.
* **`--matcher`** *(Optional)*: The feature matching algorithm to use (`SIFT`, `SuperPoint`, or `ORB`). If omitted, defaults to `SIFT`.
* **Configuration Overrides** *(Optional)*: Pipeline thresholds can be overridden at runtime using flags such as `--downscale_factor`, `--ransac_threshold`, or `--apply_clahe` / `--no-apply_clahe`. For the exhaustive list of overrides, run `python main.py --help` or refer to [`config.py`](./config.py).

## Outputs

Artifacts are automatically saved to `<DATA_DIR>/<dataset>/output/`:

* **Point Clouds:** A standard ASCII `.ply` file containing the triangulated 3D coordinates and extracted RGB colors.
* **Metrics:** A `.json` file detailing performance metrics, frame-by-frame initialization statistics, and overall reprojection errors (RMSE).

## Companion Repositories

This project operates alongside two companion repositories that complete the reconstruction ecosystem:

* **[`incremental-sfm-datasets`](https://github.com/chuo-v/incremental-sfm-datasets)**: Houses the curated image sequence datasets and intrinsic camera matrices (`K.txt`) utilized by the data loader.
* **[`incremental-sfm-experiments`](https://github.com/chuo-v/incremental-sfm-experiments)**: Contains extended experimental setups, benchmark evaluations, and analytical scripts that build upon the core incremental engine to evaluate mapping performance and feature matching robustness.