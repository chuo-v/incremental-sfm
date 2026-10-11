import logging
import time
import numpy as np

from config import config
from sfm_core.data_loader import SfMDataLoader
from sfm_core.geometry import GeometryEngine
from sfm_core.orb_matcher import ORBMatcher
from sfm_core.sift_matcher import SIFTMatcher
from sfm_core.superpoint_matcher import SuperPointMatcher

logger = logging.getLogger(__name__)


class IncrementalSfMEngine:
    """
    Orchestrates the Incremental Structure from Motion (SfM) pipeline.
    Manages data loading, feature matching, geometric tracking, global relocalization,
    3D mapping, and performance metric collection across a multi-view dataset.
    """

    def __init__(self, dataset_name: str, matcher_type: str = "SIFT"):
        """
        Initializes the SfM engine with a specified dataset and feature matcher.

        Args:
            dataset_name (str): The folder name of the dataset residing in DATA_DIR.
            matcher_type (str): The feature matching algorithm to use ('SIFT', 'ORB', 'SuperPoint').
        """
        self.data_loader = SfMDataLoader(dataset_name)
        self.K = self.data_loader.K

        if matcher_type == "SuperPoint":
            self.matcher = SuperPointMatcher()
        elif matcher_type == "ORB":
            self.matcher = ORBMatcher()
        else:
            self.matcher = SIFTMatcher()

        self.point_cloud = []
        self.point_colors = []
        self.camera_poses = {}
        self.camera_projections = {}
        self.image_cache = {}

        # Local tracker for high-speed sequential matching
        self.last_frame_2d_to_3d: dict[tuple[float, float], int] = {}
        # Global Track Manager: Maps (camera_index, (x,y)) -> 3D point index
        self.obs2d_to_3d: dict[tuple[int, tuple[float, float]], int] = {}

        self.frame_rmses = []
        self.overall_rmse = 0.0

        # Structured run metrics
        self.metrics = {"initialization": {}, "frames": [], "summary": {}}

    def _get_cached_image(self, index: int):
        """
        Loads and caches images into RAM to accelerate multi-view matching.
        Enforces a bounded cache size using LRU eviction to prevent Out-Of-Memory (OOM) errors.
        Args:
            index (int): The sequential index of the image to load.
        Returns:
            tuple: A tuple containing the RGB image array and grayscale image array.
        """
        if index in self.image_cache:
            val = self.image_cache.pop(index)
            self.image_cache[index] = val
            return val

        # Enforce cache limit by removing the oldest untouched key
        if len(self.image_cache) >= config.max_cache_size:
            oldest_key = next(iter(self.image_cache))
            del self.image_cache[oldest_key]

        self.image_cache[index] = self.data_loader.get_image(index)
        return self.image_cache[index]

    def initialize_two_view(self) -> bool:
        """
        Computes the initial two-view seed to bootstrap the 3D map.
        Extracts features between the first two images, computes the essential matrix,
        recovers the initial relative pose, and triangulates the base point cloud.

        Returns:
            bool: True if two-view initialization was successful, False otherwise.
        """
        start_time = time.perf_counter()
        logger.info("Initializing Two-View Seed (Images 0 and 1)...")

        rgb0, gray0 = self._get_cached_image(0)
        _, gray1 = self._get_cached_image(1)

        pts0, pts1 = self.matcher.match(gray0, gray1, idx1=0, idx2=1)
        if len(pts0) < 8:
            return False

        E, mask = GeometryEngine.compute_essential_matrix(pts0, pts1, self.K)
        if E.shape != (3, 3):
            return False

        mask = mask.flatten() == 1
        pts0, pts1 = pts0[mask], pts1[mask]

        R1, t1 = GeometryEngine.recover_pose(E, pts0, pts1, self.K)

        # Define Camera 0 at the origin
        R0 = np.eye(3, dtype=np.float32)
        t0 = np.zeros((3, 1), dtype=np.float32)
        self.camera_poses[0] = (R0, t0)
        self.camera_projections[0] = self.K @ np.hstack((R0, t0))
        self.camera_poses[1] = (R1, t1)
        self.camera_projections[1] = self.K @ np.hstack((R1, t1))

        pts3D, valid_mask = GeometryEngine.triangulate_and_filter(
            pts0, pts1, self.K, R0, t0, R1, t1
        )

        valid_pts0 = pts0[valid_mask]
        valid_pts1 = pts1[valid_mask]

        # Vectorized key hashing for the map population
        keys_0 = np.round(valid_pts0, 4)
        keys_1 = np.round(valid_pts1, 4)

        # Populate the initial map and track features
        for i in range(len(pts3D)):
            pt3d = pts3D[i]
            pt2d_0 = valid_pts0[i]

            x, y = int(pt2d_0[0]), int(pt2d_0[1])
            color = (
                rgb0[y, x]
                if 0 <= y < rgb0.shape[0] and 0 <= x < rgb0.shape[1]
                else np.array([0, 0, 0])
            )

            self.point_cloud.append(pt3d)
            self.point_colors.append(color)
            point_idx = len(self.point_cloud) - 1

            key_0 = (keys_0[i, 0], keys_0[i, 1])
            key_1 = (keys_1[i, 0], keys_1[i, 1])

            self.last_frame_2d_to_3d[key_1] = point_idx
            self.obs2d_to_3d[(0, key_0)] = point_idx
            self.obs2d_to_3d[(1, key_1)] = point_idx

        # Calculate and log the baseline RMSE for the seed
        rmse = GeometryEngine.calculate_reprojection_error(
            pts3D, valid_pts1, R1, t1, self.K
        )
        self.frame_rmses.append(rmse)

        elapsed_time = round(time.perf_counter() - start_time, 3)
        self.metrics["initialization"] = {
            "seeded_points": len(self.point_cloud),
            "rmse": rmse,
            "time_seconds": elapsed_time,
        }

        logger.info(
            f"Two-View Initialization complete ({elapsed_time}s). Seeded {len(self.point_cloud)} 3D points."
        )
        return True

    def _track_sequential(self, pts_prev: np.ndarray, pts_curr: np.ndarray):
        """
        Performs fast sequential N to N-1 tracking and pose estimation.
        Associates features from the current frame to the previous frame's 3D points
        and computes the camera pose using Perspective-n-Point (PnP).

        Args:
            pts_prev (np.ndarray): Matched 2D points in the previous image.
            pts_curr (np.ndarray): Matched 2D points in the current image.

        Returns:
            tuple: Tracking data including 3D-2D associations, unassociated points,
                   the computed rotation matrix, translation vector, and PnP inlier mask.
        """
        obj_pts, img_pts, associated_3d_indices = [], [], []
        unassociated_prev, unassociated_curr = [], []

        # Vectorized rounding to bypass function overhead during map lookups
        keys_prev = np.round(pts_prev, 4)

        for j in range(len(pts_prev)):
            pt_p, pt_c = pts_prev[j], pts_curr[j]
            key_p = (keys_prev[j, 0], keys_prev[j, 1])

            if key_p in self.last_frame_2d_to_3d:
                pt3d_idx = self.last_frame_2d_to_3d[key_p]
                obj_pts.append(self.point_cloud[pt3d_idx])
                img_pts.append(pt_c)
                associated_3d_indices.append(pt3d_idx)
            else:
                unassociated_prev.append(pt_p)
                unassociated_curr.append(pt_c)

        R_curr, t_curr, inlier_mask = np.empty((0, 0)), np.empty((0, 0)), np.empty(0)

        if len(obj_pts) >= 6:
            R_curr, t_curr, inlier_mask = GeometryEngine.solve_pnp(
                np.array(obj_pts), np.array(img_pts), self.K
            )

        return (
            obj_pts,
            img_pts,
            associated_3d_indices,
            unassociated_prev,
            unassociated_curr,
            R_curr,
            t_curr,
            inlier_mask,
        )

    def _relocalize_global(
        self,
        curr_idx: int,
        gray_curr: np.ndarray,
        seq_img_pts: list,
        seq_assoc_3d_indices: list,
        last_success_idx: int,
    ):
        """
        Attempts global tracking recovery when sequential tracking fails.
        Matches the current image against all previously registered images to find
        enough 3D-2D associations to compute a valid PnP pose.

        Args:
            curr_idx (int): The sequence index of the current image.
            gray_curr (np.ndarray): The grayscale array of the current image.
            seq_img_pts (list): 2D points from the failed sequential tracking attempt.
            seq_assoc_3d_indices (list): 3D point indices corresponding to seq_img_pts.
            last_success_idx (int): The index of the last successfully registered image.

        Returns:
            tuple: A boolean indicating success, followed by the recovered rotation matrix,
                   translation vector, inliers, associations, and multi-view unassociated tracks.
        """
        logger.warning(
            f"Sequential tracking lost at Image {curr_idx}. Triggering Global Relocalization..."
        )

        curr_to_3d_multi = {}
        curr_unassoc_multi = {}

        for prev_idx in self.camera_poses.keys():
            if prev_idx == last_success_idx:
                continue

            _, g_prev = self._get_cached_image(prev_idx)
            p_prev, p_curr = self.matcher.match(
                g_prev, gray_curr, idx1=prev_idx, idx2=curr_idx
            )

            if len(p_prev) < 8:
                continue

            E, ep_mask = GeometryEngine.compute_essential_matrix(p_prev, p_curr, self.K)
            if E.shape != (3, 3):
                continue

            ep_mask = ep_mask.flatten() == 1
            valid_prev, valid_curr = p_prev[ep_mask], p_curr[ep_mask]

            # Vectorized key calculations for the cross-camera checks
            keys_prev = np.round(valid_prev, 4)
            keys_curr = np.round(valid_curr, 4)

            for j in range(len(valid_prev)):
                pt_p, pt_c = valid_prev[j], valid_curr[j]
                key_p = (keys_prev[j, 0], keys_prev[j, 1])
                key_c = (keys_curr[j, 0], keys_curr[j, 1])

                obs_key = (prev_idx, key_p)
                if obs_key in self.obs2d_to_3d:
                    if key_c not in curr_to_3d_multi:
                        curr_to_3d_multi[key_c] = (pt_c, self.obs2d_to_3d[obs_key])
                else:
                    if key_c not in curr_unassoc_multi:
                        curr_unassoc_multi[key_c] = {"pt_c": pt_c, "track": []}
                    curr_unassoc_multi[key_c]["track"].append((prev_idx, pt_p))

        # Inject surviving associations from the failed sequential attempt
        keys_seq = np.round(seq_img_pts, 4) if len(seq_img_pts) > 0 else []
        for j in range(len(seq_img_pts)):
            key_seq = (keys_seq[j, 0], keys_seq[j, 1])
            curr_to_3d_multi[key_seq] = (
                seq_img_pts[j],
                seq_assoc_3d_indices[j],
            )

        R_curr, t_curr, inlier_mask = np.empty((0, 0)), np.empty((0, 0)), np.empty(0)
        obj_pts_multi, img_pts_multi, associated_3d_indices = [], [], []

        if len(curr_to_3d_multi) >= 6:
            for _, (exact_pt_c, pt3d_idx) in curr_to_3d_multi.items():
                img_pts_multi.append(exact_pt_c)
                obj_pts_multi.append(self.point_cloud[pt3d_idx])
                associated_3d_indices.append(pt3d_idx)

            R_curr, t_curr, inlier_mask = GeometryEngine.solve_pnp(
                np.array(obj_pts_multi), np.array(img_pts_multi), self.K
            )

            if R_curr.size > 0:
                logger.info(
                    f"Relocalization successful. Recovered {len(inlier_mask)} global inliers."
                )

        return (
            R_curr.size > 0,
            R_curr,
            t_curr,
            inlier_mask,
            obj_pts_multi,
            img_pts_multi,
            associated_3d_indices,
            curr_unassoc_multi,
        )

    def _triangulate_2_view(
        self,
        curr_idx: int,
        last_success_idx: int,
        unassociated_prev: list,
        unassociated_curr: list,
        rgb_curr: np.ndarray,
        new_2d_to_3d: dict,
    ) -> int:
        """
        Executes fast epipolar triangulation strictly against the previous anchor frame.
        Utilizes vectorized reprojection error calculation to batch-filter outliers
        before mapping the surviving points and extracting their colors.

        Args:
            curr_idx (int): The current image index.
            last_success_idx (int): The previous anchor image index.
            unassociated_prev (list): 2D points in the previous image not yet mapped to 3D.
            unassociated_curr (list): Corresponding 2D points in the current image.
            rgb_curr (np.ndarray): The RGB array of the current image for color extraction.
            new_2d_to_3d (dict): Dictionary to track new 2D point to 3D index mappings.

        Returns:
            int: The number of valid 3D points successfully triangulated and mapped.
        """
        num_triangulated = 0
        if len(unassociated_prev) < 8:
            return num_triangulated

        unassoc_p = np.array(unassociated_prev)
        unassoc_c = np.array(unassociated_curr)

        # Retrieve established anchor poses
        R_prev, t_prev = self.camera_poses[last_success_idx]
        R_curr, t_curr = self.camera_poses[curr_idx]

        pts3D_new, valid_mask_new = GeometryEngine.triangulate_and_filter(
            unassoc_p,
            unassoc_c,
            self.K,
            R_prev,
            t_prev,
            R_curr,
            t_curr,
        )

        valid_unassoc_p = unassoc_p[valid_mask_new]
        valid_unassoc_c = unassoc_c[valid_mask_new]

        # Vectorized Reprojection: Calculate all errors in two bulk passes
        errs_prev = GeometryEngine.calculate_reprojection_errors(
            pts3D_new, valid_unassoc_p, R_prev, t_prev, self.K
        )
        errs_curr = GeometryEngine.calculate_reprojection_errors(
            pts3D_new, valid_unassoc_c, R_curr, t_curr, self.K
        )

        # Average the errors and apply a boolean mask for points that pass the threshold
        mean_errs = (errs_prev + errs_curr) / 2.0
        ransac_mask = mean_errs <= config.ransac_threshold

        # Filter the arrays simultaneously
        final_pts3D = pts3D_new[ransac_mask]
        final_pts2D_p = valid_unassoc_p[ransac_mask]
        final_pts2D_c = valid_unassoc_c[ransac_mask]

        # Vectorized key calculations for the final insertion loop
        keys_p = np.round(final_pts2D_p, 4)
        keys_c = np.round(final_pts2D_c, 4)

        # Only loop over the surviving points for mapping and color extraction
        for k in range(len(final_pts3D)):
            pt3d = final_pts3D[k]
            pt2d_c = final_pts2D_c[k]

            x, y = int(pt2d_c[0]), int(pt2d_c[1])
            color = (
                rgb_curr[y, x]
                if 0 <= y < rgb_curr.shape[0] and 0 <= x < rgb_curr.shape[1]
                else np.array([0, 0, 0])
            )

            self.point_cloud.append(pt3d)
            self.point_colors.append(color)
            pt3d_idx = len(self.point_cloud) - 1

            key_p = (keys_p[k, 0], keys_p[k, 1])
            key_c = (keys_c[k, 0], keys_c[k, 1])

            new_2d_to_3d[key_c] = pt3d_idx
            self.obs2d_to_3d[(last_success_idx, key_p)] = pt3d_idx
            self.obs2d_to_3d[(curr_idx, key_c)] = pt3d_idx

            # Increment valid triangulated count
            num_triangulated += 1

        return num_triangulated

    def _triangulate_n_view(
        self,
        curr_idx: int,
        curr_unassoc_multi: dict,
        rgb_curr: np.ndarray,
        new_2d_to_3d: dict,
    ) -> int:
        """
        Performs heavy Direct Linear Transform (DLT) triangulation utilizing accumulated
        feature tracks across multiple camera views.

        Args:
            curr_idx (int): The current image index.
            curr_unassoc_multi (dict): Dictionary of unassociated 2D points and historical tracks.
            rgb_curr (np.ndarray): The RGB array of the current image.
            new_2d_to_3d (dict): Dictionary to track new 2D point to 3D index mappings.

        Returns:
            int: The number of valid multi-view 3D points successfully triangulated and mapped.
        """
        num_triangulated = 0

        for key_c, data in curr_unassoc_multi.items():
            pt_c = data["pt_c"]
            track = data["track"]

            # Pull precomputed P matrices instead of manual pose reconstructions
            P_matrices = [
                self.camera_projections[prev_idx] for prev_idx, _ in track
            ] + [self.camera_projections[curr_idx]]
            pts2d = [pt_p for _, pt_p in track] + [pt_c]

            pt3d = GeometryEngine.triangulate_n_views(P_matrices, pts2d)

            valid = True
            X_homog = np.append(pt3d, 1.0)

            # The 3rd row of P computes Z in camera space.
            # Reject points that fall behind any camera (cheirality check).
            for P in P_matrices:
                if P[2, :] @ X_homog <= 0:
                    valid = False
                    break

            if valid:
                mean_err = GeometryEngine.calculate_multi_view_reprojection_error(
                    pt3d, P_matrices, pts2d
                )
                if mean_err < config.ransac_threshold:
                    x, y = int(pt_c[0]), int(pt_c[1])
                    color = (
                        rgb_curr[y, x]
                        if 0 <= y < rgb_curr.shape[0] and 0 <= x < rgb_curr.shape[1]
                        else np.array([0, 0, 0])
                    )

                    self.point_cloud.append(pt3d)
                    self.point_colors.append(color)
                    pt3d_idx = len(self.point_cloud) - 1

                    new_2d_to_3d[key_c] = pt3d_idx
                    self.obs2d_to_3d[(curr_idx, key_c)] = pt3d_idx

                    for prev_idx, pt_p in track:
                        key_p = (round(float(pt_p[0]), 4), round(float(pt_p[1]), 4))
                        self.obs2d_to_3d[(prev_idx, key_p)] = pt3d_idx

                    num_triangulated += 1

        return num_triangulated

    def run(self):
        """
        Executes the main incremental SfM pipeline loop.
        Initializes the map seed, sequentially registers new images, triggers global
        relocalization upon tracking failure, maps new 3D points, and compiles performance metrics.
        """
        pipeline_start_time = time.perf_counter()

        if not self.initialize_two_view():
            logger.error("Failed to initialize two-view seed. Pipeline aborted.")
            return

        num_images = len(self.data_loader)
        frame_times = []
        # Track the last successfully registered frame
        last_success_idx = 1

        for i in range(2, num_images):
            frame_start_time = time.perf_counter()
            logger.info(f"Processing Image {i} / {num_images - 1}...")

            rgb_curr, gray_curr = self._get_cached_image(i)
            rgb_prev, gray_prev = self._get_cached_image(last_success_idx)

            pts_prev, pts_curr = self.matcher.match(
                gray_prev, gray_curr, idx1=last_success_idx, idx2=i
            )

            # -------------------------------------------------------------
            # TRACKING
            # -------------------------------------------------------------
            (
                obj_pts,
                img_pts,
                associated_3d_indices,
                unassociated_prev,
                unassociated_curr,
                R_curr,
                t_curr,
                inlier_mask,
            ) = self._track_sequential(pts_prev, pts_curr)

            # -------------------------------------------------------------
            # GLOBAL RELOCALIZATION (Triggered on Failure)
            # -------------------------------------------------------------
            relocalized = False
            curr_unassoc_multi = {}

            if R_curr.size == 0:
                (
                    relocalized,
                    R_curr,
                    t_curr,
                    inlier_mask,
                    obj_pts,
                    img_pts,
                    associated_3d_indices,
                    curr_unassoc_multi,
                ) = self._relocalize_global(
                    i, gray_curr, img_pts, associated_3d_indices, last_success_idx
                )

            if R_curr.size == 0:
                logger.warning(f"PnP failed for image {i}. Registration skipped.")
                continue

            # -------------------------------------------------------------
            # MAPPING
            # -------------------------------------------------------------
            self.camera_poses[i] = (R_curr, t_curr)
            self.camera_projections[i] = self.K @ np.hstack((R_curr, t_curr))

            inlier_obj_pts = np.array(obj_pts)[inlier_mask]
            inlier_img_pts = np.array(img_pts)[inlier_mask]

            rmse = GeometryEngine.calculate_reprojection_error(
                inlier_obj_pts, inlier_img_pts, R_curr, t_curr, self.K
            )
            self.frame_rmses.append(rmse)

            # Set up the tracker for the newly registered image (preserving surviving PnP inliers)
            new_2d_to_3d = {}
            keys_img_pts = np.round(img_pts, 4) if len(img_pts) > 0 else []

            for inlier_idx in inlier_mask:
                pt3d_idx = associated_3d_indices[inlier_idx]
                key_c = (keys_img_pts[inlier_idx, 0], keys_img_pts[inlier_idx, 1])

                new_2d_to_3d[key_c] = pt3d_idx
                self.obs2d_to_3d[(i, key_c)] = pt3d_idx

            if not relocalized:
                num_triangulated = self._triangulate_2_view(
                    i,
                    last_success_idx,
                    unassociated_prev,
                    unassociated_curr,
                    rgb_curr,
                    new_2d_to_3d,
                )
            else:
                num_triangulated = self._triangulate_n_view(
                    i, curr_unassoc_multi, rgb_curr, new_2d_to_3d
                )

            # -------------------------------------------------------------
            # LOOP TEARDOWN
            # -------------------------------------------------------------
            self.last_frame_2d_to_3d = new_2d_to_3d
            last_success_idx = i

            frame_elapsed = round(time.perf_counter() - frame_start_time, 3)
            frame_times.append(frame_elapsed)

            logger.info(
                f"Registered Image {i} ({frame_elapsed}s): {len(inlier_mask)} PnP Inliers | {num_triangulated} Triangulated Points"
            )

            self.metrics["frames"].append(
                {
                    "image_index": i,
                    "raw_matches": len(pts_curr),
                    "associations": len(obj_pts),
                    "pnp_inliers": len(inlier_mask),
                    "triangulated_points": num_triangulated,
                    "rmse": rmse,
                    "time_seconds": frame_elapsed,
                    "is_relocalized": relocalized,
                }
            )

        total_elapsed = round(time.perf_counter() - pipeline_start_time, 3)
        self.overall_rmse = (
            float(np.mean(self.frame_rmses)) if self.frame_rmses else 0.0
        )
        avg_frame_time = round(float(np.mean(frame_times)), 3) if frame_times else 0.0

        self.metrics["summary"] = {
            "total_3d_points": len(self.point_cloud),
            "overall_rmse": self.overall_rmse,
            "total_time_seconds": total_elapsed,
            "avg_frame_time_seconds": avg_frame_time,
        }

        logger.info("=" * 30)
        logger.info(
            f"Incremental SfM Pipeline complete in {total_elapsed}s.\n"
            f"Total 3D points: {len(self.point_cloud)} | "
            f"Overall RMSE: {self.overall_rmse:.3f} px | "
            f"Avg Frame Time: {avg_frame_time:.3f}s"
        )
