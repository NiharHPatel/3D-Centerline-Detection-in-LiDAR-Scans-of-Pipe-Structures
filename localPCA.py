import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points), pcd


def estimate_global_axis_pca(points, line_length=40, resolution=300):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, Vt = np.linalg.svd(points_centered)
    main_axis = Vt[0]

    # Generate points along PCA axis
    t = np.linspace(-line_length / 2, line_length / 2, resolution)
    centerline = centroid + np.outer(t, main_axis)
    return centerline


def visualize_centerline_with_pointcloud(raw_points, centerline, title="Centerline"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*raw_points.T, s=0.5, alpha=0.08, color='gray')
    ax.plot(*centerline.T, color='green', linewidth=2, label='Centerline')
    ax.set_title(title)
    ax.legend()
    plt.show()


def process_file_pca(ply_path):
    raw_points, _ = load_point_cloud(ply_path)
    axis_pts = estimate_global_axis_pca(raw_points)
    visualize_centerline_with_pointcloud(raw_points, axis_pts, title=ply_path.split("/")[-1])
    return axis_pts


# 🔁 Replace with your actual 9 .ply file paths
ply_paths = [
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
]

# 🔄 Process each file
for path in ply_paths:
    print(f"\n📂 Processing: {path}")
    process_file_pca(path)
