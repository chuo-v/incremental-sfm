from config import config
from sfm_core.data_loader import SfMDataLoader


def test_data_loader_paths_and_scaling(tmp_path, monkeypatch):
    """Tests image sorting and intrinsic matrix (K) downscaling logic."""
    # Setup a dummy dataset structure in a temporary test directory
    dataset_dir = tmp_path / "dummy_dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)

    # Create dummy image files out of order to test sorting
    (images_dir / "02.jpg").touch()
    (images_dir / "01.jpg").touch()

    # Create a dummy K.txt file
    k_file = dataset_dir / "K.txt"
    k_file.write_text("100.0 0.0 50.0\n0.0 100.0 50.0\n0.0 0.0 1.0")

    # Inject test configurations using monkeypatch
    monkeypatch.setattr(config, "data_dir", tmp_path)
    monkeypatch.setattr(config, "downscale_factor", 0.5)

    # Initialize the loader
    loader = SfMDataLoader("dummy_dataset")

    # Verify standard image sorting
    assert len(loader) == 2
    assert loader.image_paths[0].name == "01.jpg"

    # Verify intrinsic matrix scaling
    # If downscale is 0.5, focal lengths (100) and optical centers (50) must halve, but Z scale (1.0) remains
    assert loader.K[0, 0] == 50.0
    assert loader.K[0, 2] == 25.0
    assert loader.K[2, 2] == 1.0
