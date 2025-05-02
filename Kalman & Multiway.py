import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN

# Load the LiDAR point cloud
pcd = o3d.io.read_point_cloud("/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply")
points = np.asarray(pcd.points)

# STEP 1: Fit a Cylinder using RANSAC
ransac = RANSACRegressor()
ransac.fit(points[:, :2], points[:, 2])  # Fit a cylindrical model

# Extract inliers (best-fit points for the cylinder)
inliers = points[ransac.inlier_mask_]

# Save the fitted cylinder point cloud
fitted_pcd = o3d.geometry.PointCloud()
fitted_pcd.points = o3d.utility.Vector3dVector(inliers)
o3d.io.write_point_cloud("fitted_cylinder.ply", fitted_pcd)

# STEP 2: Extract the Centerline from the Cylinder Fit
cylinder_axis = np.mean(inliers, axis=0)  # Compute Cylinder Axis
centerline_points = np.column_stack((
    np.full(100, cylinder_axis[0]),  # X remains constant
    np.full(100, cylinder_axis[1]),  # Y remains constant
    np.linspace(min(inliers[:,2]), max(inliers[:,2]), 100)  # Z varies
))

# Save the extracted centerline
np.savetxt("extracted_centerline.txt", centerline_points)

# STEP 3: Optimize Cylinder Radius Dynamically
distances = np.linalg.norm(inliers - cylinder_axis, axis=1)
optimal_radius = np.mean(distances)  # Compute optimal radius
print(f"Optimized Cylinder Radius: {optimal_radius}")

# STEP 4: Detect Bends and High-Error Regions
threshold = optimal_radius * 1.2  # Define error threshold
high_error_points = inliers[distances > threshold]  # Extract points with high deviation

# Apply DBSCAN Clustering to Segment Bends
clustering = DBSCAN(eps=optimal_radius, min_samples=5).fit(high_error_points)
bend_labels = clustering.labels_

# Save the segmented high-error (bent) regions
bend_pcd = o3d.geometry.PointCloud()
bend_pcd.points = o3d.utility.Vector3dVector(high_error_points)
o3d.io.write_point_cloud("bent_regions.ply", bend_pcd)

# STEP 5: Visualize Results
o3d.visualization.draw_geometries([fitted_pcd, bend_pcd])

print("✅ Processing Complete: Cylinder Fitted, Centerline Extracted, Bends Detected")
