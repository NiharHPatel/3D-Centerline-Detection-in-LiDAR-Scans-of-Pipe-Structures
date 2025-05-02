import open3d as o3d
import numpy as np
import os

# List of centerline PLY file paths
centerline_files = [
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

def align_centerlines_with_icp(source, target):
    """ Use ICP to align centerlines from different frames. """
    threshold = 0.02  # Distance threshold for ICP
    trans_init = np.eye(4)  # Initial transformation

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    return reg_p2p.transformation

def multiway_registration(centerlines):
    """ Align multiple centerlines using Pose Graph Optimization. """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    accum_transform = np.eye(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(accum_transform))

    for i in range(len(centerlines) - 1):
        source = centerlines[i]
        target = centerlines[i + 1]

        transformation = align_centerlines_with_icp(source, target)
        accum_transform = np.dot(accum_transform, transformation)

        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(accum_transform))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, i + 1, transformation, information=np.eye(6), uncertain=False
            )
        )

        # Show each alignment step
        transformed_source = source.transform(transformation)
        print(f"🔹 Showing alignment for step {i + 1}")
        o3d.visualization.draw_geometries([target, transformed_source])

    return pose_graph

# Load centerline point clouds
if len(centerline_files) < 2:
    print(" Not enough centerline files for Multiway Registration.")
else:
    centerlines = [o3d.io.read_point_cloud(file) for file in centerline_files]

    # Perform multiway registration
    pose_graph = multiway_registration(centerlines)

    # Apply transformations
    print("Final Visualization: All Transformed Centerlines")
    for i, centerline in enumerate(centerlines):
        centerline.transform(pose_graph.nodes[i].pose)
        o3d.visualization.draw_geometries([centerline])

    print(" Multiway registration complete.")
