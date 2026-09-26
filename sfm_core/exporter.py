import numpy as np


def export_point_cloud_ply(
    filename: str, points: list[np.ndarray], colors: list[np.ndarray]
) -> None:
    """
    Exports a 3D point cloud with colors to a standard ASCII PLY file.
    """
    if not points:
        print("Warning: Point cloud is empty. No PLY file generated.")
        return

    print(f"Exporting {len(points)} points to {filename}...")

    with open(filename, "w") as f:
        # Write PLY Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write Vertex Data
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            # Ensure colors are cast to standard 8-bit unsigned integers for the PLY format
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print("Export complete.")
