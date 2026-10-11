import cv2
import numpy as np

from config import config


class GeometryEngine:
    """
    Handles the core mathematical operations for the Incremental SfM pipeline.
    """

    @staticmethod
    def compute_essential_matrix(
        pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Essential Matrix from two sets of matched 2D points using RANSAC.

        Args:
            pts1: (N, 2) array of points in Image 1.
            pts2: (N, 2) array of corresponding points in Image 2.
            K: 3x3 camera intrinsic matrix.

        Returns:
            E: 3x3 Essential Matrix.
            mask: (N, 1) array indicating inliers (1) and outliers (0) from RANSAC.
        """
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        # Calculate the geometric relationship between two cameras
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=config.ransac_prob,
            threshold=config.ransac_threshold,
        )

        # If insufficient inliers exist, RANSAC fails and returns None
        if E is None:
            return np.empty((0, 0)), np.empty((0, 0))

        # OpenCV Edge Case: If points are co-planar or ambiguous, findEssentialMat
        # can occasionally return multiple stacked 3x3 matrices as a (9, 3) array.
        # For incremental SfM, strictly isolate the first valid 3x3 solution.
        if E.shape != (3, 3):
            E = E[0:3, 0:3]

        return E, mask

    @staticmethod
    def recover_pose(
        E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decomposes the Essential Matrix to recover the relative camera rotation and translation.

        Args:
            E: 3x3 Essential Matrix.
            pts1: (N, 2) array of points in Image 1.
            pts2: (N, 2) array of corresponding points in Image 2.
            K: 3x3 camera intrinsic matrix.

        Returns:
            R: 3x3 rotation matrix.
            t: 3x1 translation vector.
        """
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
        return R, t

    @staticmethod
    def solve_pnp(
        obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perspective-n-Point (PnP): Calculates camera pose (R, t) from 3D-2D correspondences.

        Args:
            obj_pts: (N, 3) array of 3D object points.
            img_pts: (N, 2) array of corresponding 2D image observations.
            K: 3x3 camera intrinsic matrix.

        Returns:
            R: 3x3 rotation matrix of the camera pose. Returns empty array on failure.
            t: 3x1 translation vector of the camera pose. Returns empty array on failure.
            inlier_mask: 1D boolean/integer array indicating PnP RANSAC inliers.
        """
        obj_pts = np.float32(obj_pts)
        img_pts = np.float32(img_pts)

        success, rvec, t, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=100,
            reprojectionError=config.ransac_threshold,
            confidence=config.ransac_prob,
        )

        if not success or inliers is None:
            return np.empty((0, 0)), np.empty((0, 0)), np.empty(0)

        R, _ = cv2.Rodrigues(rvec)
        inlier_mask = inliers.flatten()

        return R, t, inlier_mask

    @staticmethod
    def triangulate_and_filter(
        pts1: np.ndarray,
        pts2: np.ndarray,
        K: np.ndarray,
        R1: np.ndarray,
        t1: np.ndarray,
        R2: np.ndarray,
        t2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Triangulates 2D points into 3D space and applies a strict Cheirality check
        dynamically against any two camera positions.
        Safely handles division-by-zero for points at infinity.

        Args:
            pts1: (N, 2) array of matched points in Image 1.
            pts2: (N, 2) array of corresponding matched points in Image 2.
            K: 3x3 camera intrinsic matrix.
            R1: 3x3 rotation matrix for Camera 1.
            t1: 3x1 translation vector for Camera 1.
            R2: 3x3 rotation matrix for Camera 2.
            t2: 3x1 translation vector for Camera 2.

        Returns:
            valid_pts3D: (M, 3) array of valid 3D points that passed all geometric filters.
            full_mask: (N,) boolean mask indicating which original point pairs survived.
        """
        if len(pts1) == 0:
            return np.empty((0, 3)), np.zeros(0, dtype=bool)

        # Construct Projection Matrices
        Rt1 = np.hstack((R1, t1))
        P1 = K @ Rt1
        Rt2 = np.hstack((R2, t2))
        P2 = K @ Rt2

        # Triangulate Points
        pts1_t = np.float32(pts1).T
        pts2_t = np.float32(pts2).T

        # Returns points in 4D homogeneous coordinates: (4, N)
        pts4D = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)

        # Safe Homogeneous to 3D Cartesian Conversion
        w = pts4D[3, :]
        # Filter out rays that are perfectly parallel (points at infinity) where W == 0
        valid_w = np.abs(w) > 1e-7

        if not np.any(valid_w):
            return np.empty((0, 3)), np.zeros(len(pts1), dtype=bool)

        # Divide strictly the valid points to prevent RuntimeWarnings and NaNs
        pts3D = pts4D[:3, valid_w] / w[valid_w]  # Shape: (3, M)

        # Aggressive Cheirality Check (Z > 0)
        pts3D_cam1 = (R1 @ pts3D) + t1
        z_cam1 = pts3D_cam1[2, :] > 0.0

        pts3D_cam2 = (R2 @ pts3D) + t2
        z_cam2 = pts3D_cam2[2, :] > 0.0

        # Minimum Triangulation Angle (Parallax) Check
        C1 = -R1.T @ t1
        C2 = -R2.T @ t2

        ray1 = pts3D - C1
        ray2 = pts3D - C2

        norm1 = np.linalg.norm(ray1, axis=0, keepdims=True)
        norm2 = np.linalg.norm(ray2, axis=0, keepdims=True)

        ray1_norm = ray1 / (norm1 + 1e-7)
        ray2_norm = ray2 / (norm2 + 1e-7)

        cos_angle = np.sum(ray1_norm * ray2_norm, axis=0)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180.0 / np.pi)

        # Require minimum degrees of parallax
        parallax_mask = angle > config.min_parallax_degrees

        # Combine masks to aggressively filter out invalid points
        cheirality_mask = z_cam1 & z_cam2 & parallax_mask

        # Map the surviving points back to the original N-length boolean mask
        full_mask = np.zeros(len(pts1), dtype=bool)
        valid_indices = np.where(valid_w)[0][cheirality_mask]
        full_mask[valid_indices] = True

        valid_pts3D = pts3D[:, cheirality_mask].T  # Transpose back to (K, 3)
        return valid_pts3D, full_mask

    @staticmethod
    def triangulate_n_views(
        poses: list[tuple[np.ndarray, np.ndarray]],
        pts2d: list[np.ndarray],
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Multi-View Triangulation using Direct Linear Transform (DLT).
        Mathematically intersects rays from N cameras using SVD to reduce depth uncertainty.

        Args:
            poses: List of (R, t) tuples representing the poses of N cameras.
            pts2d: List of (2,) arrays representing the 2D coordinates in each camera view.
            K: 3x3 camera intrinsic matrix.

        Returns:
            X: (3,) array representing the triangulated 3D point in Cartesian coordinates.
               Returns a zero array if the point is at infinity.
        """
        A = []
        for (R, t), pt in zip(poses, pts2d):
            P = K @ np.hstack((R, t))
            x, y = pt[0], pt[1]

            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])

        A = np.array(A)

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        if np.abs(X[3]) < 1e-7:
            return np.zeros(3)

        return X[:3] / X[3]

    @staticmethod
    def calculate_reprojection_error(
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
    ) -> float:
        """
        Calculates the Root Mean Square Error (RMSE) between actual 2D observations
        and the 3D points projected back onto the image plane.

        Args:
            obj_pts: (N, 3) array of triangulated 3D points.
            img_pts: (N, 2) array of observed 2D image points.
            R: 3x3 camera rotation matrix.
            t: 3x1 camera translation vector.
            K: 3x3 camera intrinsic matrix.

        Returns:
            float: The calculated RMSE in pixels. Returns 0.0 if no points are provided.
        """
        if len(obj_pts) == 0:
            return 0.0

        rvec, _ = cv2.Rodrigues(R)

        # Project the 3D points back to 2D
        projected_pts, _ = cv2.projectPoints(
            np.float32(obj_pts), rvec, np.float32(t), K, None
        )
        projected_pts = projected_pts.reshape(-1, 2)

        errors = np.linalg.norm(np.float32(img_pts) - projected_pts, axis=1)

        return float(np.sqrt(np.mean(errors**2)))

    @staticmethod
    def calculate_reprojection_errors(
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Efficiently calculates individual reprojection errors for a batch of points.

        Args:
            obj_pts: (N, 3) array of triangulated 3D points.
            img_pts: (N, 2) array of observed 2D image points.
            R: 3x3 camera rotation matrix.
            t: 3x1 camera translation vector.
            K: 3x3 camera intrinsic matrix.

        Returns:
            np.ndarray: An (N,) array of reprojection errors in pixels.
        """
        if len(obj_pts) == 0:
            return np.empty(0)

        rvec, _ = cv2.Rodrigues(R)
        projected_pts, _ = cv2.projectPoints(
            np.float32(obj_pts), rvec, np.float32(t), K, None
        )
        projected_pts = projected_pts.reshape(-1, 2)

        # Return the L2 distance (error in pixels) for each point
        return np.linalg.norm(np.float32(img_pts) - projected_pts, axis=1)

    @staticmethod
    def calculate_multi_view_reprojection_error(
        pt3d: np.ndarray,
        poses: list[tuple[np.ndarray, np.ndarray]],
        pts2d: list[np.ndarray],
        K: np.ndarray,
    ) -> float:
        """
        Efficiently calculates the Root Mean Square Error (RMSE) of a 3D point
        projected across multiple camera poses using direct pinhole projection
        (assuming zero distortion).
        """
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X = np.asarray(pt3d, dtype=np.float32).reshape(3, 1)
        squared_errors = []
        for (R, t), pt2d in zip(poses, pts2d):
            X_cam = R @ X + t
            z = X_cam[2, 0]
            if z <= 0:
                return float("inf")
            u = fx * (X_cam[0, 0] / z) + cx
            v = fy * (X_cam[1, 0] / z) + cy
            dx = u - pt2d[0]
            dy = v - pt2d[1]
            squared_errors.append(dx * dx + dy * dy)
        if not squared_errors:
            return 0.0
        return float(np.sqrt(np.mean(squared_errors)))
