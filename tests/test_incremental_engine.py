from unittest.mock import MagicMock
from config import config
from sfm_core.incremental_engine import IncrementalSfMEngine


def test_image_caching_and_eviction(monkeypatch):
    """Verifies that the engine's image_cache strictly follows LRU eviction."""
    monkeypatch.setattr(config, "max_cache_size", 2)

    # Mock the dataloader to prevent actual disk reads
    mock_data_loader = MagicMock()
    mock_data_loader.get_image.side_effect = lambda idx: (f"rgb_{idx}", f"gray_{idx}")
    mock_data_loader.K = MagicMock()

    # Intercept SfMDataLoader initialization so the engine uses our mock
    monkeypatch.setattr(
        "sfm_core.incremental_engine.SfMDataLoader", lambda x: mock_data_loader
    )

    engine = IncrementalSfMEngine(dataset_name="dummy_dataset", matcher_type="SIFT")

    # 1. Cache idx=0 (Cache Miss)
    engine._get_cached_image(0)
    assert 0 in engine.image_cache

    # 2. Cache idx=1 (Cache Miss)
    engine._get_cached_image(1)
    assert len(engine.image_cache) == 2

    # 3. Access idx=0 again (Cache Hit)
    # This promotes idx=0 to Most Recently Used. idx=1 is now Least Recently Used.
    engine._get_cached_image(0)

    # 4. Cache idx=2 (Cache Miss)
    # Triggers eviction of the Least Recently Used entry (idx 1).
    engine._get_cached_image(2)

    # Verify strict LRU eviction
    assert 1 not in engine.image_cache  # Evicted because it was Least Recently Used
    assert 0 in engine.image_cache  # Retained because it was recently accessed
    assert 2 in engine.image_cache  # Retained because it was just added

    # Ensure get_image was only called 3 times (the cache hit on step 3 shouldn't call it)
    assert mock_data_loader.get_image.call_count == 3
