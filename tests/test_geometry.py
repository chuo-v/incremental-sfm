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


def test_calculate_reprojection_errors_vectorized():
    """Verifies that the vectorized reprojection error calculates correct L2 distances."""
    # Simple camera setup at origin looking down +Z
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    R = np.eye(3, dtype=np.float32)
    t = np.zeros((3, 1), dtype=np.float32)

    # 3D points at Z=10
    obj_pts = np.array(
        [
            [0.0, 0.0, 10.0],  # Projects to (50, 50)
            [1.0, 1.0, 10.0],  # Projects to (60, 60)
        ],
        dtype=np.float32,
    )

    # 2D observations with known pixel errors
    # Point 1: Observed exactly at (50, 50) -> Error = 0
    # Point 2: Observed at (63, 64) -> Expected at (60, 60) -> Error = sqrt(3^2 + 4^2) = 5
    img_pts = np.array(
        [
            [50.0, 50.0],
            [63.0, 64.0],
        ],
        dtype=np.float32,
    )

    errors = GeometryEngine.calculate_reprojection_errors(obj_pts, img_pts, R, t, K)

    assert errors.shape == (2,)
    assert np.isclose(errors[0], 0.0)
    assert np.isclose(errors[1], 5.0)


def test_calculate_reprojection_errors_empty():
    """Tests the failsafe when passing empty arrays to the vectorized error calculator."""
    K = np.eye(3, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros((3, 1), dtype=np.float32)

    errors = GeometryEngine.calculate_reprojection_errors(
        np.empty((0, 3)), np.empty((0, 2)), R, t, K
    )

    assert errors.size == 0
    assert isinstance(errors, np.ndarray)


def test_triangulate_n_views():
    """Verifies multi-view triangulation (DLT) across 3 cameras."""
    # Simple intrinsic matrix
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    # Camera 1: Origin
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Camera 2: Shifted by 2 units on X axis (t = [-2, 0, 0]^T)
    P2 = K @ np.hstack((np.eye(3), np.array([[-2.0], [0.0], [0.0]])))

    # Camera 3: Shifted by 2 units on Y axis (t = [0, -2, 0]^T)
    P3 = K @ np.hstack((np.eye(3), np.array([[0.0], [-2.0], [0.0]])))

    P_matrices = [P1, P2, P3]

    # 3D Point at [0, 0, 10] projects to:
    # Cam 1: [50, 50]
    # Cam 2: [30, 50]
    # Cam 3: [50, 30]
    pts2d = [np.array([50.0, 50.0]), np.array([30.0, 50.0]), np.array([50.0, 30.0])]

    pt3d = GeometryEngine.triangulate_n_views(P_matrices, pts2d)

    assert pt3d.shape == (3,)
    assert np.allclose(pt3d, [0.0, 0.0, 10.0], atol=1e-5)


def test_calculate_multi_view_reprojection_error():
    """Verifies RMSE calculation across multiple projection matrices."""
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((np.eye(3), np.array([[-2.0], [0.0], [0.0]])))
    P_matrices = [P1, P2]

    pt3d = np.array([0.0, 0.0, 10.0])

    # Expected projections: [50, 50] and [30, 50]
    # Introduce an error of 3px (X) and 4px (Y) for Cam 1 -> Euclidean Distance = 5px, Squared = 25
    # Cam 2 is perfectly observed -> Distance = 0px, Squared = 0
    # Mean Squared Error = 25 / 2 = 12.5
    pts2d = [np.array([53.0, 54.0]), np.array([30.0, 50.0])]

    error = GeometryEngine.calculate_multi_view_reprojection_error(
        pt3d, P_matrices, pts2d
    )

    expected_rmse = np.sqrt(12.5)
    assert np.isclose(error, expected_rmse)


def test_calculate_multi_view_reprojection_error_behind_camera():
    """Verifies that points falling behind a camera (cheirality failure) return infinity."""
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Point behind the camera (Z = -10)
    pt3d = np.array([0.0, 0.0, -10.0])
    pts2d = [np.array([50.0, 50.0])]

    error = GeometryEngine.calculate_multi_view_reprojection_error(pt3d, [P1], pts2d)
    assert error == float("inf")


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
