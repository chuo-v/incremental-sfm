# incremental-sfm

`incremental-sfm` is a modular, pure-Python Incremental Structure from Motion (SfM) pipeline that orchestrates sequential data loading, polymorphic feature matching, epipolar initialization, Perspective-n-Point (PnP) geometric tracking, global relocalization, and multi-view 3D mapping. It processes ordered multi-view image sequences and exports colored 3D point clouds (`.ply`) alongside detailed frame-by-frame telemetry (`.json`).

## Companion Repositories

This project operates alongside two companion repositories that complete the reconstruction ecosystem:

- **[`incremental-sfm-datasets`](https://github.com/chuo-v/incremental-sfm-datasets):** Houses the curated NTU campus image sequence datasets (`nantah-memorial-P42` and `yunnan-garden-pavilion-P71`) and calibrated camera intrinsic matrices (`K.txt`).
- **[`incremental-sfm-experiments`](https://github.com/chuo-v/incremental-sfm-experiments):** Contains evaluation notebooks, ablation studies, post-processing outlier removal scripts, quantitative telemetry logs, and interactive 3D HTML visualizations.

## Features

- **Polymorphic Feature Matchers:** Supports classical handcrafted extractors (`SIFT` with optional `RootSIFT` normalization, and `ORB`) alongside deep learning-based `SuperPoint` paired with `LightGlue` (with automatic hardware acceleration across NVIDIA `cuda`, Apple Silicon `mps`, and `cpu`).
- **Robust Tracking & Global Relocalization:** Implements fast sequential $N \to N-1$ tracking via EPnP RANSAC (`SOLVEPNP_EPNP`) with automatic global relocalization against historical keyframes (with Essential Matrix RANSAC epipolar gating) whenever sequential tracking is lost.
- **Dual-Mode Dynamic Triangulation:** Utilizes fast two-view epipolar triangulation during sequential tracking and switches to multi-view Singular Value Decomposition Direct Linear Transform (SVD-DLT) across accumulated feature tracks during global relocalization, enforcing strict dual/multi-camera cheirality ($Z > 0$) and minimum ray-parallax filtering.
- **Bounded LRU Memory Caching:** Implements two independent Least-Recently-Used (LRU) caches (`image_cache` and `feature_cache`) bounded by `max_cache_size` to eliminate redundant disk I/O and descriptor re-extraction during multi-view relocalization while preventing Out-Of-Memory (OOM) crashes.
- **Configurable Pipeline:** Exposes all image processing, feature extraction, and geometric RANSAC hyperparameters via [`config.py`](./config.py) and runtime CLI flags.

## Repository Structure

```text
incremental-sfm/
├── sfm_core/
│   ├── __init__.py
│   ├── data_loader.py          # Sequential image loading, downscaling, and K matrix scaling
│   ├── feature_matcher.py      # Abstract BaseFeatureMatcher contract, LRU cache, and bidirectional KNN
│   ├── sift_matcher.py         # OpenCV SIFT / RootSIFT extractor + FLANN KD-Tree matcher
│   ├── orb_matcher.py          # OpenCV ORB binary extractor + Brute-Force Hamming matcher
│   ├── superpoint_matcher.py   # PyTorch SuperPoint extractor + LightGlue transformer matcher
│   ├── geometry.py             # Essential Matrix RANSAC, EPnP, 2-view & N-view DLT triangulation
│   ├── incremental_engine.py   # Core SfM state machine, track manager, and telemetry logger
│   └── exporter.py             # ASCII PLY 3D point cloud exporter
├── tests/
│   ├── test_data_loader.py     # Unit tests for path sorting and intrinsic matrix scaling
│   ├── test_geometry.py        # Unit tests for cheirality, zero-W safety, DLT, and reprojection error
│   ├── test_incremental_engine.py # Unit tests for engine-level LRU image caching
│   └── test_matcher.py         # Unit tests for SIFT, SuperPoint, ORB, and feature caching
├── .env.example                # Template for configuring the local or Colab DATA_DIR path
├── config.py                   # Central PipelineConfig dataclass and .env loader
├── main.py                     # CLI entry point and argument parser
├── pyproject.toml              # Package build configuration, dependencies, pytest, and ruff settings
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

## Installation

1. Ensure your environment runs **Python 3.10** or higher.
2. Clone this repository and install the package and its dependencies (OpenCV, NumPy, PyTorch, TorchVision, and LightGlue):
    ```bash
    pip install .
    ```
    *Note: To include development tools (`pytest`, `pytest-mock`, and `ruff`), install in editable mode with:*
    ```bash
    pip install -e ".[dev]"
    ```
3. Configure your dataset directory by copying [`.env.example`](./.env.example) to `.env` in the project root and setting `DATA_DIR` to an existing directory on your machine:
    ```bash
    cp .env.example .env
    # Or set directly:
    echo "DATA_DIR=/path/to/your/datasets" > .env
    ```

## Dataset Structure

Each dataset must be placed inside your configured `DATA_DIR` and adhere to the following structure:

```text
<DATA_DIR>/<dataset_folder_name>/
├── images/      # Sequential image files (.jpg, .jpeg, .png), sorted alphabetically
└── K.txt        # A 3x3 space-separated camera intrinsic matrix
```

## Usage

Execute the pipeline via the command line using `main.py`:

```bash
python main.py --dataset <dataset_folder_name> --matcher <MATCHER_TYPE> [OPTIONS]
```

### Example Commands

```bash
# Example 1: Run a standard dataset at full resolution with SIFT
python main.py --dataset my_custom_dataset --matcher SIFT --max-features 50000 --min-parallax-degrees 1.0

# Example 2: Run a high-resolution outdoor dataset at half resolution with SuperPoint + LightGlue
python main.py \
    --dataset large_outdoor_sequence \
    --matcher SuperPoint \
    --downscale-factor 0.5 \
    --max-cache-size 100 \
    --max-features 50000 \
    --apply-clahe \
    --ransac-threshold 1.5 \
    --min-parallax-degrees 1.0
```

### CLI Arguments & Configuration Overrides

All optional flags override the default values defined in [`config.py`](./config.py):

| Argument | Type / Choices | Default (`config.py`) | Description |
| --- | --- | --- | --- |
| `--dataset` | `str` *(Required)* | — | Name of the dataset folder inside `DATA_DIR`. |
| `--matcher` | `SIFT`, `SuperPoint`, `ORB` | `SIFT` | Feature extraction and matching backend to use. |
| `--downscale-factor` | `float` | `1.0` | Image resolution multiplier (automatically scales $f_x, f_y, c_x, c_y$ in $K$). |
| `--max-cache-size` | `int` | `15` | Maximum items retained in the LRU `image_cache` and `feature_cache`. |
| `--max-features` | `int` | `8192` | Maximum keypoints extracted per image across all matchers. |
| `--lowe-ratio-threshold` | `float` | `0.75` | Lowe's ratio test threshold for filtering ambiguous KNN matches (`SIFT` & `ORB`). |
| `--apply-clahe` / `--no-apply-clahe` | `bool` flag | `True` | Enables or disables CLAHE (`clipLimit=2.0`, `8x8` grid) prior to extraction. |
| `--disable-symmetric-matching` | `bool` flag | `False` | Disables bidirectional cross-check matching to halve KNN matching time (`SIFT` & `ORB`). |
| `--use-root-sift` / `--no-use-root-sift` | `bool` flag | `False` | Converts SIFT descriptors to RootSIFT ($L_1$ normalization + square root; `SIFT` only). |
| `--ransac-prob` | `float` | `0.999` | RANSAC confidence probability for Essential Matrix and EPnP solvers. |
| `--ransac-threshold` | `float` | `3.0` | Maximum reprojection error (in pixels) for RANSAC inliers and triangulation filtering. |
| `--min-parallax-degrees` | `float` | `1.5` | Minimum camera ray intersection angle (in degrees) required to validate a 3D point. |

## Outputs

Artifacts are automatically exported to `<DATA_DIR>/<dataset>/output/`:

* **Point Clouds (`output/point_clouds/point_cloud_<MATCHER>.ply`):** A standard ASCII `.ply` file containing the triangulated 3D coordinates (`x, y, z`) and extracted RGB vertex colors (`red, green, blue`).
* **Telemetry Metrics (`output/metrics/run_metrics_<MATCHER>.json`):** A structured `.json` log recording the active configuration, two-view seed initialization statistics, per-frame tracking and triangulation counts (`raw_matches`, `associations`, `pnp_inliers`, `triangulated_points`, `rmse`, `time_seconds`, `is_relocalized`), and overall run summary metrics (`total_3d_points`, `overall_rmse`, `total_time_seconds`, `avg_frame_time_seconds`).

## Testing & Code Quality

Run the unit test suite (`pytest`) and linter (`ruff`) from the repository root (note that `DATA_DIR` must point to an existing directory via `.env` or an inline environment variable so `config.py` initializes cleanly):

```bash
# Execute all unit tests (uses .env if configured, or pass an existing path inline)
DATA_DIR=. pytest

# Check code formatting and linting
ruff check .
```