import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

# -----------------------------
# User: Manually list your 9 PLY file paths
# -----------------------------
ply_files = [
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

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_point_cloud(pcd, voxel_size=0.01):
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    return pcd

# -----------------------------
# Skeleton Extraction
# -----------------------------
def build_knn_graph(points, k=8):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    return distances, indices

def extract_skeleton(points):
    distances, indices = build_knn_graph(points)
    adjacency = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in indices[i]:
            if i != j:
                adjacency[i, j] = np.linalg.norm(points[i] - points[j])
    mst = minimum_spanning_tree(adjacency).toarray()
    rows, cols = np.where(mst > 0)
    skeleton_edges = np.array([[r, c] for r, c in zip(rows, cols)])
    return skeleton_edges

# -----------------------------
# Skeleton Ordering
# -----------------------------
def order_skeleton(points, edges):
    ordered = [edges[0][0], edges[0][1]]
    edges = list(edges[1:])
    while edges:
        last = ordered[-1]
        for i, (u, v) in enumerate(edges):
            if u == last:
                ordered.append(v)
                edges.pop(i)
                break
            elif v == last:
                ordered.append(u)
                edges.pop(i)
                break
        else:
            break
    return points[ordered]

# -----------------------------
# B-Spline Fitting
# -----------------------------
def smooth_spline(points, s=1.0):
    tck, u = splprep(points.T, s=s, k=3)
    u_fine = np.linspace(0, 1, 500)
    smoothed = np.array(splev(u_fine, tck)).T
    return smoothed

# -----------------------------
# Batch Process and Visualization
# -----------------------------
for file in ply_files:
    print(f"\nProcessing: {os.path.basename(file)}")

    pcd = o3d.io.read_point_cloud(file)
    pcd = preprocess_point_cloud(pcd)

    points = np.asarray(pcd.points)
    edges = extract_skeleton(points)
    ordered_points = order_skeleton(points, edges)
    smoothed_centerline = smooth_spline(ordered_points, s=0.5)

    # Visualize before saving
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, label='Point Cloud')
    ax.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], c='r', label='Centerline')
    ax.legend()
    plt.title(f"{os.path.basename(file)}")
    plt.show()

print("\n✅ Batch processing and visualization complete.")
