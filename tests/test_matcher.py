from unittest.mock import MagicMock, patch
import numpy as np
import torch

from config import config
from sfm_core.sift_matcher import SIFTMatcher
from sfm_core.orb_matcher import ORBMatcher
from sfm_core.superpoint_matcher import SuperPointMatcher


# ---------------------------------------------------------
# SIFT Tests
# ---------------------------------------------------------
def test_sift_matcher_empty_image_failsafe():
    """Tests the safety check when feature descriptors cannot be extracted."""
    matcher = SIFTMatcher()
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)
    pts1, pts2 = matcher.match(img1, img2)
    assert pts1.shape == (0, 2)
    assert pts2.shape == (0, 2)


def test_sift_matcher_lowes_ratio():
    matcher = SIFTMatcher()
    matcher.sift = MagicMock()
    matcher.matcher = MagicMock()
    kp1 = [MagicMock(pt=(10.0, 10.0)), MagicMock(pt=(20.0, 20.0))]
    kp2 = [MagicMock(pt=(15.0, 15.0)), MagicMock(pt=(25.0, 25.0))]
    match1_fwd = [
        MagicMock(distance=0.5, queryIdx=0, trainIdx=0),
        MagicMock(distance=1.0),
    ]
    match2_fwd = [
        MagicMock(distance=0.8, queryIdx=1, trainIdx=1),
        MagicMock(distance=1.0),
    ]
    match1_bwd = [
        MagicMock(distance=0.5, queryIdx=0, trainIdx=0),
        MagicMock(distance=1.0),
    ]
    match2_bwd = [
        MagicMock(distance=0.8, queryIdx=1, trainIdx=1),
        MagicMock(distance=1.0),
    ]
    matcher.sift.detectAndCompute.side_effect = [
        (kp1, np.ones((2, 128))),
        (kp2, np.ones((2, 128))),
    ]
    matcher.matcher.knnMatch.side_effect = [
        [match1_fwd, match2_fwd],
        [match1_bwd, match2_bwd],
    ]
    img_dummy = np.zeros((100, 100), dtype=np.uint8)
    pts1, pts2 = matcher.match(img_dummy, img_dummy)
    assert len(pts1) == 1
    assert len(pts2) == 1
    assert np.array_equal(pts1[0], [10.0, 10.0])
    assert np.array_equal(pts2[0], [15.0, 15.0])


def test_sift_matcher_symmetric_cross_check():
    matcher = SIFTMatcher()
    matcher.sift = MagicMock()
    matcher.matcher = MagicMock()
    kp1 = [MagicMock(pt=(10.0, 10.0))]
    kp2 = [MagicMock(pt=(15.0, 15.0)), MagicMock(pt=(20.0, 20.0))]
    m_fwd = [MagicMock(distance=0.1, queryIdx=0, trainIdx=0), MagicMock(distance=1.0)]
    m_bwd = [MagicMock(distance=0.1, queryIdx=0, trainIdx=5), MagicMock(distance=1.0)]
    matcher.sift.detectAndCompute.side_effect = [
        (kp1, np.ones((1, 128))),
        (kp2, np.ones((2, 128))),
    ]
    matcher.matcher.knnMatch.side_effect = [[m_fwd], [m_bwd]]
    img_dummy = np.zeros((100, 100), dtype=np.uint8)
    pts1, pts2 = matcher.match(img_dummy, img_dummy)
    assert len(pts1) == 0
    assert len(pts2) == 0


def test_sift_matcher_clahe_applied(monkeypatch):
    monkeypatch.setattr(config, "apply_clahe", True)
    matcher = SIFTMatcher()
    matcher.clahe = MagicMock()
    matcher.clahe.apply.return_value = np.zeros((10, 10), dtype=np.uint8)
    matcher.sift = MagicMock()
    matcher.sift.detectAndCompute.return_value = (
        [MagicMock(pt=(0, 0)), MagicMock(pt=(1, 1))],
        np.ones((2, 128)),
    )
    matcher.matcher = MagicMock()
    img_dummy = np.zeros((100, 100), dtype=np.uint8)
    matcher.match(img_dummy, img_dummy)
    assert matcher.clahe.apply.call_count == 2


# ---------------------------------------------------------
# SuperPoint Tests
# ---------------------------------------------------------
@patch("sfm_core.superpoint_matcher.LightGlue")
@patch("sfm_core.superpoint_matcher.SuperPoint")
def test_superpoint_preprocess_image(mock_sp, mock_lg):
    """Verifies that images are correctly converted to normalized PyTorch tensors."""
    matcher = SuperPointMatcher()

    # Dummy 10x10 image with max uint8 value
    img = np.full((100, 100), 255, dtype=np.uint8)
    tensor = matcher._preprocess_image(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 100, 100)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.ones((1, 1, 100, 100), device=matcher.device))


@patch("sfm_core.superpoint_matcher.rbd")
@patch("sfm_core.superpoint_matcher.LightGlue")
@patch("sfm_core.superpoint_matcher.SuperPoint")
def test_superpoint_empty_matches(mock_sp, mock_lg, mock_rbd):
    """Tests the failsafe when LightGlue finds zero matches between images."""
    matcher = SuperPointMatcher()

    # The rbd (remove batch dimension) utility acts as a pass-through in this mock
    mock_rbd.side_effect = lambda x: x

    # Overwrite the initialized PyTorch models with standard MagicMocks
    matcher.extractor = MagicMock()
    matcher.matcher = MagicMock()

    # Simulate a network output where no matches survive
    matcher.matcher.return_value = {"matches": []}

    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)

    pts1, pts2 = matcher.match(img1, img2)

    assert pts1.shape == (0, 2)
    assert pts2.shape == (0, 2)


@patch("sfm_core.superpoint_matcher.rbd")
@patch("sfm_core.superpoint_matcher.LightGlue")
@patch("sfm_core.superpoint_matcher.SuperPoint")
def test_superpoint_successful_match(mock_sp, mock_lg, mock_rbd):
    """Verifies that tensor outputs are correctly extracted, indexed, and pushed to NumPy."""
    matcher = SuperPointMatcher()
    mock_rbd.side_effect = lambda x: x
    matcher.extractor = MagicMock()
    matcher.matcher = MagicMock()

    # Mock SuperPoint extractor to return dummy keypoint tensors
    kp_tensor = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    matcher.extractor.extract.side_effect = [
        {"keypoints": kp_tensor},
        {"keypoints": kp_tensor},
    ]

    # Mock LightGlue to match Point 0 to Point 1, and Point 2 to Point 2
    matcher.matcher.return_value = {"matches": torch.tensor([[0, 1], [2, 2]])}

    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)

    pts1, pts2 = matcher.match(img1, img2)

    assert pts1.shape == (2, 2)
    assert pts2.shape == (2, 2)

    assert np.array_equal(pts1[0], [10.0, 10.0])
    assert np.array_equal(pts2[0], [20.0, 20.0])

    assert np.array_equal(pts1[1], [30.0, 30.0])
    assert np.array_equal(pts2[1], [30.0, 30.0])


@patch("torch.cuda.is_available", return_value=True)
@patch("sfm_core.superpoint_matcher.LightGlue")
@patch("sfm_core.superpoint_matcher.SuperPoint")
def test_superpoint_device_cuda(mock_sp, mock_lg, mock_cuda):
    """Verifies that CUDA is prioritized if available."""
    matcher = SuperPointMatcher()
    assert matcher.device.type == "cuda"


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
@patch("sfm_core.superpoint_matcher.LightGlue")
@patch("sfm_core.superpoint_matcher.SuperPoint")
def test_superpoint_device_mps(mock_sp, mock_lg, mock_mps, mock_cuda):
    """Verifies that MPS (Apple Silicon) is selected if CUDA is unavailable."""
    matcher = SuperPointMatcher()
    assert matcher.device.type == "mps"


# ---------------------------------------------------------
# ORB Tests
# ---------------------------------------------------------
def test_orb_matcher_empty_image_failsafe():
    """Tests the safety check when ORB feature descriptors cannot be extracted."""
    matcher = ORBMatcher()

    # Pass completely black, featureless 10x10 images
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)

    pts1, pts2 = matcher.match(img1, img2)

    # The matcher should catch the lack of descriptors and return empty (0, 2) arrays
    assert pts1.shape == (0, 2)
    assert pts2.shape == (0, 2)


def test_orb_matcher_lowes_ratio():
    """Verifies that Lowe's Ratio Test correctly filters ambiguous matches using ORB."""
    matcher = ORBMatcher()

    matcher.orb = MagicMock()
    matcher.matcher = MagicMock()

    # Mock keypoints
    kp1 = [MagicMock(pt=(10.0, 10.0)), MagicMock(pt=(20.0, 20.0))]
    kp2 = [MagicMock(pt=(15.0, 15.0)), MagicMock(pt=(25.0, 25.0))]

    # Mock KNN matches (k=2) for Forward Pass (Img1 -> Img2)
    # Match 1: Ratio 0.5 < (0.75 * 1.0) -> SHOULD PASS Lowe's
    match1_fwd = [
        MagicMock(distance=0.5, queryIdx=0, trainIdx=0),
        MagicMock(distance=1.0),
    ]
    # Match 2: Ratio 0.8 > (0.75 * 1.0) -> SHOULD FAIL Lowe's
    match2_fwd = [
        MagicMock(distance=0.8, queryIdx=1, trainIdx=1),
        MagicMock(distance=1.0),
    ]

    # Mock KNN matches (k=2) for Backward Pass (Img2 -> Img1) to satisfy Cross-Check
    match1_bwd = [
        MagicMock(distance=0.5, queryIdx=0, trainIdx=0),
        MagicMock(distance=1.0),
    ]
    match2_bwd = [
        MagicMock(distance=0.8, queryIdx=1, trainIdx=1),
        MagicMock(distance=1.0),
    ]

    # Inject mock returns onto the MagicMock objects (ORB uses 32-byte uint8 descriptors)
    matcher.orb.detectAndCompute.side_effect = [
        (kp1, np.ones((2, 32), dtype=np.uint8)),
        (kp2, np.ones((2, 32), dtype=np.uint8)),
    ]

    matcher.matcher.knnMatch.side_effect = [
        [match1_fwd, match2_fwd],
        [match1_bwd, match2_bwd],
    ]

    # Execute matcher on dummy images
    img_dummy = np.zeros((100, 100), dtype=np.uint8)
    pts1, pts2 = matcher.match(img_dummy, img_dummy)

    # Verify only Match 1 survived the Lowe's ratio filter and the cross-check
    assert len(pts1) == 1
    assert len(pts2) == 1
    assert np.array_equal(pts1[0], [10.0, 10.0])
    assert np.array_equal(pts2[0], [15.0, 15.0])


def test_orb_matcher_symmetric_cross_check():
    """Verifies that an asymmetric match is strictly rejected by the ORBMatcher."""
    matcher = ORBMatcher()

    matcher.orb = MagicMock()
    matcher.matcher = MagicMock()

    kp1 = [MagicMock(pt=(10.0, 10.0))]
    kp2 = [MagicMock(pt=(15.0, 15.0)), MagicMock(pt=(20.0, 20.0))]

    # Forward Match: Img1's Point 0 chooses Img2's Point 0 (Passes Lowe's Ratio)
    m_fwd = [MagicMock(distance=0.1, queryIdx=0, trainIdx=0), MagicMock(distance=1.0)]

    # Backward Match: Img2's Point 0 chooses a DIFFERENT point in Img1 (TrainIdx 5 instead of 0)
    m_bwd = [MagicMock(distance=0.1, queryIdx=0, trainIdx=5), MagicMock(distance=1.0)]

    matcher.orb.detectAndCompute.side_effect = [
        (kp1, np.ones((1, 32), dtype=np.uint8)),
        (kp2, np.ones((2, 32), dtype=np.uint8)),
    ]
    matcher.matcher.knnMatch.side_effect = [[m_fwd], [m_bwd]]

    img_dummy = np.zeros((100, 100), dtype=np.uint8)
    pts1, pts2 = matcher.match(img_dummy, img_dummy)

    # The cross-check must catch the mismatch and return empty arrays
    assert len(pts1) == 0
    assert len(pts2) == 0


def test_orb_matcher_clahe_applied(monkeypatch):
    """Verifies that CLAHE pre-processing runs on both images when enabled in config."""
    monkeypatch.setattr(config, "apply_clahe", True)
    matcher = ORBMatcher()

    # Mock the CLAHE object initialized inside ORBMatcher
    matcher.clahe = MagicMock()
    matcher.clahe.apply.return_value = np.zeros((10, 10), dtype=np.uint8)

    # Provide dummy ORB data to prevent crashes downstream
    matcher.orb = MagicMock()
    matcher.orb.detectAndCompute.return_value = (
        [MagicMock(pt=(0, 0)), MagicMock(pt=(1, 1))],
        np.ones((2, 32), dtype=np.uint8),
    )
    matcher.matcher = MagicMock()

    img_dummy = np.zeros((100, 100), dtype=np.uint8)
    matcher.match(img_dummy, img_dummy)

    # Verify CLAHE was applied exactly twice (once per image)
    assert matcher.clahe.apply.call_count == 2


# ---------------------------------------------------------
# Caching Tests
# ---------------------------------------------------------
def test_feature_caching_and_eviction(monkeypatch):
    """Verifies that the feature cache stores, retrieves, and evicts correctly based on config."""
    # Force a tiny cache size to easily test eviction
    monkeypatch.setattr(config, "max_cache_size", 2)
    matcher = SIFTMatcher()

    # Mock the extraction method to track how many times it actually processes an image
    matcher._extract = MagicMock(return_value=(["kp_mock"], ["des_mock"]))
    img_dummy = np.zeros((10, 10), dtype=np.uint8)

    # 1. First image (Cache Miss -> Extracted and Stored)
    matcher._get_features(img_dummy, idx=0)
    assert matcher._extract.call_count == 1
    assert 0 in matcher.feature_cache

    # 2. Same image requested again (Cache Hit -> Pulled from RAM)
    matcher._get_features(img_dummy, idx=0)
    assert matcher._extract.call_count == 1  # Call count stays at 1

    # 3. Add second image (Cache Miss -> Extracted and Stored)
    matcher._get_features(img_dummy, idx=1)
    assert matcher._extract.call_count == 2
    assert len(matcher.feature_cache) == 2

    # 4. Add third image (Triggers eviction of the oldest entry: idx 0)
    matcher._get_features(img_dummy, idx=2)
    assert matcher._extract.call_count == 3
    assert len(matcher.feature_cache) == 2
    assert 0 not in matcher.feature_cache
    assert 1 in matcher.feature_cache
    assert 2 in matcher.feature_cache
