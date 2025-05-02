import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os

# === CONFIG ===
voxel_size = 0.02
radius = 0.5
min_neighbors = 5
spline_smoothness = 1.0
spline_points = 300

# === FUNCTIONS ===
def preprocess_point_cloud(pcd):
    pcd = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd.points)
    return pts, pcd

def extract_local_centers(points, radius=0.5, min_neighbors=5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    centers = []

    for i, pt in enumerate(points):
        try:
            [_, idxs, _] = kdtree.search_radius_vector_3d(pt, radius)
        except RuntimeError:
            continue

        if len(idxs) >= min_neighbors:
            local_pts = points[idxs]
            center = np.median(local_pts, axis=0)
            centers.append(center)

    centers = np.array(centers)
    if len(centers) == 0:
        print("⚠️ Still no centerline points found.")
    else:
        print(f"✅ Found {len(centers)} center points.")

    return centers

def fit_spline(points, smooth=1.0, n=300):
    if len(points) < 4:
        return points
    sorted_points = points[np.argsort(points[:, 2])]
    tck, _ = splprep(sorted_points.T, s=smooth)
    u_fine = np.linspace(0, 1, n)
    return np.array(splev(u_fine, tck)).T

def plot_centerline(raw_points, centerline, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*raw_points.T, s=0.5, alpha=0.05, color='gray')
    ax.plot(*centerline.T, color='green', linewidth=2, label='Ali-style Centerline')
    ax.set_title(title)
    ax.legend()
    plt.show()

# === FILE PATHS ===
ply_paths = [
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
  #  "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
   # "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
]

# === RUN PIPELINE ===
for path in ply_paths:
    print(f"\n📂 Processing: {os.path.basename(path)}")
    if not os.path.exists(path):
        print("❌ File not found.")
        continue

    pcd = o3d.io.read_point_cloud(path)
    points, _ = preprocess_point_cloud(pcd)
    centers = extract_local_centers(points, radius=radius, min_neighbors=min_neighbors)

    if len(centers) > 0:
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(centers)
        temp_pcd.paint_uniform_color([1, 0, 0])  # Red for centers
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd, temp_pcd])

        centerline = fit_spline(centers, smooth=spline_smoothness, n=spline_points)
        plot_centerline(points, centerline, title=os.path.basename(path))

print("\n✅ Done with all files.")
