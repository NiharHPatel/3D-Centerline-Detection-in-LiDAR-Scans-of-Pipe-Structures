import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
import os

# List of detected centerline files
centerline_files = [
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_1_0058.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_2_0002.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_3_0000.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_4_0005.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_5_0015.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_6_0006.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_7_0029.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_8_0007.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_9_0000.000.ply"
]

def load_ply_points(file_path):
    """ Load PLY file and return points as NumPy array. """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

# Load all centerline points
all_centerlines = [load_ply_points(f) for f in centerline_files if load_ply_points(f) is not None]

# Ensure all centerlines have the same number of points (for proper averaging)
min_length = min(len(cl) for cl in all_centerlines)
trimmed_centerlines = [cl[:min_length] for cl in all_centerlines]

# Compute the average centerline
average_centerline = np.mean(trimmed_centerlines, axis=0)

# Interpolate with a B-spline for smoothness
tck, u = splprep(average_centerline.T, s=0.1, k=3)
u_fine = np.linspace(0, 1, 500)
smoothed_centerline = np.array(splev(u_fine, tck)).T

# Convert to Open3D PointCloud
expected_centerline_pcd = o3d.geometry.PointCloud()
expected_centerline_pcd.points = o3d.utility.Vector3dVector(smoothed_centerline)

# Save the expected centerline
expected_centerline_file = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/expected_centerline.ply"
o3d.io.write_point_cloud(expected_centerline_file, expected_centerline_pcd)

print(f"Expected centerline saved: {expected_centerline_file}")
