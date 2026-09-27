import logging
import numpy as np

logger = logging.getLogger(__name__)


def export_point_cloud_ply(
    filename: str, points: list[np.ndarray], colors: list[np.ndarray]
) -> None:
    """
    Exports a 3D point cloud with colors to a standard ASCII PLY file.
    """
    if not points:
        logger.warning("Point cloud is empty. No PLY file generated.")
        return

    logger.info(f"Exporting {len(points)} points to {filename}...")

    pts_arr = np.asarray(points, dtype=np.float32)
    color_arr = np.asarray(colors, dtype=np.uint8)

    data = np.hstack((pts_arr, color_arr))

    # Define the standard PLY header
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header"
    )

    np.savetxt(
        filename,
        data,
        fmt="%f %f %f %d %d %d",
        header=header,
        comments="",  # Prevent savetxt from prepending '#' to the header lines
    )

    logger.info("Export complete.")
