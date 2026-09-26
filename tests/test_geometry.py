from unittest.mock import patch
import numpy as np

from config import config
from sfm_core.geometry import GeometryEngine


def test_compute_essential_matrix_insufficient_points():
    """Tests the failsafe when RANSAC does not have enough points to compute a matrix."""
    # 3 points are insufficient for findEssentialMat (requires 5+)
    pts1 = np.array([[0, 0], [1, 1], [2, 2]])
    pts2 = np.array([[0, 0], [1, 1], [2, 2]])
    K = np.eye(3, dtype=np.float32)

    E, mask = GeometryEngine.compute_essential_matrix(pts1, pts2, K)

    assert E.size == 0
    assert mask.size == 0


def test_compute_essential_matrix_stacked_matrices():
    """Tests the OpenCV edge case where findEssentialMat returns a (9, 3) array."""
    # Mock a 9x3 matrix (3 stacked 3x3 matrices)
    stacked_E = np.vstack(
        [np.eye(3, dtype=np.float32), np.ones((3, 3)), np.zeros((3, 3))]
    )

    with patch("cv2.findEssentialMat", return_value=(stacked_E, np.ones((10, 1)))):
        pts1 = np.ones((10, 2))
        pts2 = np.ones((10, 2))
        K = np.eye(3, dtype=np.float32)

        E, mask = GeometryEngine.compute_essential_matrix(pts1, pts2, K)

        # The engine must slice out only the first 3x3 matrix to avoid crashing recoverPose
        assert E.shape == (3, 3)
        assert np.array_equal(E, np.eye(3, dtype=np.float32))


@patch("cv2.triangulatePoints")
def test_triangulate_and_filter_cheirality(mock_triangulate, monkeypatch):
    """Mocks triangulation to strictly test the Z>0 cheirality filter."""
    # Isolate the test from global config changes
    monkeypatch.setattr(config, "min_parallax_degrees", 1.5)

    K = np.eye(3, dtype=np.float32)
    R1, R2 = np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    # Offset t2 so the cameras have a baseline, creating ~11.3 deg of parallax
    t1 = np.zeros((3, 1), dtype=np.float32)
    t2 = np.array([[-1.0], [0.0], [0.0]], dtype=np.float32)

    # Mock the 4D homogeneous points normally returned by cv2.triangulatePoints
    # Point 1: Z=5.0 (Valid, in front of camera)
    # Point 2: Z=-2.0 (Invalid, behind camera)
    mock_triangulate.return_value = np.array(
        [[0.0, 0.0], [0.0, 0.0], [5.0, -2.0], [1.0, 1.0]]
    )

    # Dummy 2D points
    pts1 = np.zeros((2, 2))
    pts2 = np.zeros((2, 2))

    pts3D, mask = GeometryEngine.triangulate_and_filter(pts1, pts2, K, R1, t1, R2, t2)

    # Verify that the filter successfully stripped the point with a negative Z value
    assert len(pts3D) == 1
    assert pts3D[0, 2] == 5.0
    assert np.array_equal(mask, [True, False])


@patch("cv2.triangulatePoints")
def test_triangulate_and_filter_zero_w(mock_triangulate, monkeypatch):
    """Verifies that points with near-zero or zero W coordinates are filtered safely without warnings."""
    # Isolate the test from global config changes
    monkeypatch.setattr(config, "min_parallax_degrees", 1.5)

    K = np.eye(3, dtype=np.float32)
    R1, R2 = np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    # Offset t2 so the cameras have a baseline, creating ~11.3 deg of parallax
    t1 = np.zeros((3, 1), dtype=np.float32)
    t2 = np.array([[-1.0], [0.0], [0.0]], dtype=np.float32)

    # Point 1: W=0.0 (Invalid, at infinity / division by zero)
    # Point 2: W=1.0, Z=5.0 (Valid, in front of camera)
    mock_triangulate.return_value = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 5.0], [0.0, 1.0]]
    )

    pts1 = np.zeros((2, 2))
    pts2 = np.zeros((2, 2))
    pts3D, mask = GeometryEngine.triangulate_and_filter(pts1, pts2, K, R1, t1, R2, t2)

    assert len(pts3D) == 1
    assert pts3D[0, 2] == 5.0
    assert np.array_equal(mask, [False, True])


@patch("cv2.solvePnPRansac")
def test_solve_pnp_failure_handling(mock_solve_pnp):
    """Tests graceful failure when PnP cannot find a valid pose (e.g., lost tracking)."""
    # Mock solvePnPRansac to simulate a failure to find a pose (success = False)
    mock_solve_pnp.return_value = (False, None, None, None)

    obj_pts = np.ones((10, 3))
    img_pts = np.ones((10, 2))
    K = np.eye(3, dtype=np.float32)

    R, t, mask = GeometryEngine.solve_pnp(obj_pts, img_pts, K)

    # Must return empty arrays so the incremental loop can skip the frame safely
    assert R.size == 0
    assert t.size == 0
    assert mask.size == 0
