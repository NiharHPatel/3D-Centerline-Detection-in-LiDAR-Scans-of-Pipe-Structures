import open3d as o3d
import numpy as np
import os
import csv
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors

# === PATH CONFIGURATION ===
original_ply_dir = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/"
extracted_centerline_dir = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULTIWAY/"
expected_centerline_path = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULTIWAY/expected_centerline.ply"
output_dir = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Validation Results/"

os.makedirs(output_dir, exist_ok=True)

# === UTILITY FUNCTIONS ===
def load_ply(file_path):
    return o3d.io.read_point_cloud(file_path)

def chamfer_distance(A, B):
    nn_A = NearestNeighbors(n_neighbors=1).fit(B)
    dist_A, _ = nn_A.kneighbors(A)
    nn_B = NearestNeighbors(n_neighbors=1).fit(A)
    dist_B, _ = nn_B.kneighbors(B)
    return np.mean(dist_A) + np.mean(dist_B)

def hausdorff_distance(A, B):
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])

# === LOAD EXPECTED CENTERLINE ===
expected_pcd = load_ply(expected_centerline_path)
expected_np = np.asarray(expected_pcd.points)

# === CSV LOG SETUP ===
csv_path = os.path.join(output_dir, "validation_metrics.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Chamfer Distance", "Hausdorff Distance"])

# === PROCESS EACH FRAME ===
for i in range(1, 10):
    frame_id = f"{i}_{'0000.000' if i == 9 else '000' + str(i).zfill(2) + '.500'}"
    original_file = os.path.join(original_ply_dir, f"{frame_id}.ply")
    centerline_file = os.path.join(extracted_centerline_dir, f"centerline_{frame_id}.ply")

    if not os.path.exists(original_file) or not os.path.exists(centerline_file):
        print(f"Skipping {frame_id} – file not found.")
        continue

    print(f"Processing Frame: {frame_id}")
    # Load data
    original_pcd = load_ply(original_file)
    centerline_pcd = load_ply(centerline_file)
    centerline_np = np.asarray(centerline_pcd.points)

    # Compute distances
    chamfer = chamfer_distance(centerline_np, expected_np)
    hausdorff = hausdorff_distance(centerline_np, expected_np)
    csv_writer.writerow([frame_id, f"{chamfer:.6f}", f"{hausdorff:.6f}"])

    # Visualize and save screenshot
    original_pcd.paint_uniform_color([0.6, 0.6, 0.6])    # Gray
    centerline_pcd.paint_uniform_color([1.0, 0.0, 0.0])   # Red
    expected_pcd.paint_uniform_color([0.0, 1.0, 0.0])     # Green

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(original_pcd)
    vis.add_geometry(centerline_pcd)
    vis.add_geometry(expected_pcd)
    vis.poll_events()
    vis.update_renderer()

    screenshot_path = os.path.join(output_dir, f"screenshot_{frame_id}.png")
    vis.capture_screen_image(screenshot_path)
    vis.destroy_window()
    print(f"✅ Saved screenshot: {screenshot_path}")

csv_file.close()
print(f"✅ All metrics saved to: {csv_path}")
