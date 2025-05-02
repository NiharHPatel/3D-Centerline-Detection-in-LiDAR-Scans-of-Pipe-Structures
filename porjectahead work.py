import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# List of PLY file paths (Update paths accordingly)
file_paths = [
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
]

# Function to process a single file
def process_single_file(file_path):
    try:
        print(f"\nProcessing: {file_path}")

        # Load individual PLY file
        pcd = o3d.io.read_point_cloud(file_path)

        # Step 1: Preprocess the point cloud
        def preprocess_point_cloud(pcd, voxel_size=0.02):
            """Downsample the point cloud for easier processing."""
            return pcd.voxel_down_sample(voxel_size=voxel_size)

        filtered_pcd = preprocess_point_cloud(pcd)

        # Step 2: Slice the point cloud along the Z-axis
        def slice_point_cloud(pcd, z_interval=0.1):
            """Slice the point cloud into sections along the Z-axis."""
            points = np.asarray(pcd.points)
            min_z, max_z = points[:, 2].min(), points[:, 2].max()
            slices = []
            for z in np.arange(min_z, max_z, z_interval):
                mask = (points[:, 2] >= z) & (points[:, 2] < z + z_interval)
                slice_points = points[mask]
                if len(slice_points) > 0:
                    slices.append(slice_points)
            return slices

        z_slices = slice_point_cloud(filtered_pcd, z_interval=0.1)

        # Step 3: Compute centroids for each slice
        slice_centroids = []
        for i, slice_points in enumerate(z_slices):
            xy_points = slice_points[:, :2]  # Only consider X and Y for convex hull
            if len(xy_points) >= 3:  # ConvexHull requires at least 3 points
                hull = ConvexHull(xy_points)
                boundary_points = xy_points[hull.vertices]
                centroid = np.mean(boundary_points, axis=0)
                slice_centroids.append([centroid[0], centroid[1], np.mean(slice_points[:, 2])])

        slice_centroids = np.array(slice_centroids)

        # Step 4: Fit a 3D spline through the centroids
        def fit_centerline(centroids):
            """Fit a 3D spline through the centroids."""
            tck, u = splprep(centroids.T, s=1)  # Smoothness parameter `s`
            u_fine = np.linspace(0, 1, 500)
            return np.array(splev(u_fine, tck)).T

        centerline = fit_centerline(slice_centroids)

        # Step 5: Visualize the centerline and the point cloud
        centerline_pcd = o3d.geometry.PointCloud()
        centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
        centerline_pcd.paint_uniform_color([1, 0, 0])  # Red color for centerline

        # Paint the filtered point cloud for visualization
        filtered_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Grey color for pipe
        o3d.visualization.draw_geometries([filtered_pcd, centerline_pcd], window_name=f"Centerline - {file_path}")

        # Step 6: Plot the centerline in 3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(slice_centroids[:, 0], slice_centroids[:, 1], slice_centroids[:, 2], c="green", label="Centroids")
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], c="red", label="Fitted Centerline")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.title(f"3D Centerline for {file_path.split('/')[-1]}")
        plt.show()

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# **Process Each File One-by-One**
for file in file_paths:
    process_single_file(file)
    input("\nPress Enter to proceed to the next file...\n")  # Wait before moving to the next file
