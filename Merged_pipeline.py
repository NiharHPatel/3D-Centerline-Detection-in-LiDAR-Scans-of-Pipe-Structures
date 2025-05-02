import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.neighbors import NearestNeighbors
from pykalman import KalmanFilter

# 📌 **Directory of PLY Files**
ply_files = [
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    #r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
]

# 📌 **Preprocessing: Denoising & Outlier Removal**
def preprocess_point_cloud(pcd, voxel_size=0.01):
    """Downsample and remove noise from the point cloud."""
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd

# 📌 **Color-Based Segmentation (Filter outliers by intensity)**
def color_based_segmentation(pcd, color_threshold=0.2):
    """Segment the pipe based on intensity (grayscale or color data)."""
    colors = np.asarray(pcd.colors)
    mask = np.linalg.norm(colors, axis=1) > color_threshold
    return pcd.select_by_index(np.where(mask)[0])

# 📌 **Extract Centerline (Improved Spline & Kalman Filter)**
def extract_centerline(pcd):
    """Compute smooth centerline using spline fitting and Kalman filtering."""
    points = np.asarray(pcd.points)
    if len(points) < 5:
        return points

    points = points[np.argsort(points[:, 2])]  # Sort along Z-axis
    tck, u = splprep(points.T, s=0.1, k=3)
    u_fine = np.linspace(0, 1, 500)
    smooth_centerline = np.array(splev(u_fine, tck)).T

    # Apply Kalman Filter for additional smoothing
    kf = KalmanFilter(initial_state_mean=smooth_centerline[0], n_dim_obs=3)
    return kf.smooth(smooth_centerline)[0]

# 📌 **Colored Point Cloud Registration (Global Registration)**
def colored_icp_registration(source, target):
    """Use Colored Point Cloud Registration before ICP."""
    voxel_radius = [0.02, 0.01, 0.005]
    current_transformation = np.eye(4)
    for radius in voxel_radius:
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=30))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=30))

        reg_p2p = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
        )
        current_transformation = reg_p2p.transformation
    return current_transformation

# 📌 **Multiway Registration (Align all pipes)**
def multiway_registration(centerlines):
    """Align multiple centerlines using pose graph optimization."""
    pose_graph = o3d.pipelines.registration.PoseGraph()
    accum_transform = np.eye(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(accum_transform))

    for i in range(len(centerlines) - 1):
        source = centerlines[i]
        target = centerlines[i + 1]

        transformation = colored_icp_registration(source, target)
        accum_transform = np.dot(accum_transform, transformation)

        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(accum_transform))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, i + 1, transformation, information=np.eye(6), uncertain=False
            )
        )

        # Show intermediate steps
        transformed_source = source.transform(transformation)
        print(f"🔹 Aligning Step {i + 1}")
        o3d.visualization.draw_geometries([target, transformed_source])

    return pose_graph

# 📌 **Load and Process Centerlines**
if len(ply_files) < 2:
    print("❌ Not enough centerline files for Multiway Registration.")
else:
    centerlines = []
    
    for file in ply_files:
        if not os.path.exists(file):
            print(f"❌ File not found: {file}")
            continue

        pcd = o3d.io.read_point_cloud(file)
        pcd_filtered = preprocess_point_cloud(pcd)
        pcd_segmented = color_based_segmentation(pcd_filtered)
        centerline = extract_centerline(pcd_segmented)

        # Convert centerline to Open3D PointCloud
        centerline_pcd = o3d.geometry.PointCloud()
        centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
        centerline_pcd.paint_uniform_color([1, 0, 0])  # Red centerline

        centerlines.append(centerline_pcd)

    if len(centerlines) < 2:
        print("❌ Not enough valid centerlines to proceed.")
    else:
        # Perform multiway registration with colored ICP
        pose_graph = multiway_registration(centerlines)

        # Apply transformations
        print("✅ Final Visualization: Aligned & Enhanced Pipeline")
        for i, centerline in enumerate(centerlines):
            centerline.transform(pose_graph.nodes[i].pose)

        # Enhanced visualization with black background
        o3d.visualization.draw_geometries(centerlines, zoom=0.8, background_color=(0, 0, 0))
