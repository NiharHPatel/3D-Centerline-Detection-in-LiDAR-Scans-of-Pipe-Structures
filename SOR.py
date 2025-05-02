import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

# Output directory for saving images and processed point clouds
output_dir = "output_sor"
os.makedirs(output_dir, exist_ok=True)

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

# Function to apply SOR 
def apply_sor(pcd, nb_neighbors=30, std_ratio=2.0):
    """
    Apply Statistical Outlier Removal (SOR).
    """
    print("Applying Statistical Outlier Removal (SOR)...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

# Process each PLY file
for file_path in file_paths:
    try:
        print(f"📌 Processing file: {file_path}")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            print(f"⚠ Warning: {file_path} contains no valid points.")
            continue

        # Apply SOR 
        pcd_sor = apply_sor(pcd)

        # Save processed point cloud
        sor_ply_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.ply', '_sor.ply')}")
        o3d.io.write_point_cloud(sor_ply_path, pcd_sor)
        print(f"✅ SOR Processed file saved: {sor_ply_path}")

        # Visualize and save the image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd_sor)
        vis.poll_events()
        vis.update_renderer()
        image_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.ply', '_sor.png')}")
        vis.capture_screen_image(image_path)
        vis.destroy_window()
        print(f"📷 Image saved: {image_path}")

        # Plot a simple point cloud visualization using Matplotlib
        points = np.asarray(pcd_sor.points)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, color="blue")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"SOR Processed - {os.path.basename(file_path)}")
        plt_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.ply', '_sor_matplotlib.png')}")
        plt.savefig(plt_path)
        plt.close()
        print(f"📷 Matplotlib image saved: {plt_path}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")




# Process each PLY file
for file_path in file_paths:
    try:
        print(f"📌 Processing file: {file_path}")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            print(f"⚠ Warning: {file_path} contains no valid points.")
            continue

        # Apply SOR (Noise Removal)
        pcd_sor = apply_sor(pcd)

        # Convert to NumPy array for Matplotlib plotting
        points = np.asarray(pcd_sor.points)

        # **3D Scatter Plot for Visualization**
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, color="blue")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"SOR Processed - {os.path.basename(file_path)}")
        plt.show()  # Show the plot for interactive visualization

    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
