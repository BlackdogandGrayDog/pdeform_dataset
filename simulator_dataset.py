import cv2
import numpy as np
import open3d as o3d
import PythonCDT as cdt
import matplotlib.pyplot as plt
import os
import numpy as np
import OpenEXR
import Imath
import re
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs_SP
from dust3r.inference import inference
from dust3r.utils.image import load_images
from scipy.spatial import cKDTree
import shutil

# Automatically detects the prefix of stereo image files
def detect_prefix(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        match = re.match(r"^([A-Z]+)_\d{5}\.png$", file)  # Match pattern like 'KOR_00001.png'
        if match:
            return match.group(1)  # Extract prefix (e.g., 'KOR')
    raise ValueError("Could not detect a valid prefix from image filenames in the folder!")

# Function to create grid points on an image for initialization
def create_grid_points(image_shape, grid_size, margin_ratio=0.1, use_random_margin=False, use_random_offset=False, use_jitter=False, jitter_std=1.5):
    H, W = image_shape[:2]
    if use_random_margin:
        margin_ratio = np.random.uniform(0.1, 0.12)

    margin_x = int(W * margin_ratio)
    margin_y = int(H * margin_ratio)

    x_coords = np.linspace(margin_x, W - margin_x, grid_size, dtype=int)
    y_coords = np.linspace(margin_y, H - margin_y, grid_size, dtype=int)

    if use_random_offset:
        stride_x = (W - 2 * margin_x) / (grid_size - 1)
        stride_y = (H - 2 * margin_y) / (grid_size - 1)
        offset_x = int(np.random.uniform(0, 0.25 * stride_x))
        offset_y = int(np.random.uniform(0, 0.25 * stride_y))
        x_coords = np.clip(x_coords + offset_x, 0, W - 1)
        y_coords = np.clip(y_coords + offset_y, 0, H - 1)

    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    if use_jitter:
        jitter = np.random.normal(scale=jitter_std, size=grid_points.shape)
        grid_points = np.clip(grid_points + jitter, [0, 0], [W - 1, H - 1]).astype(int)

    return grid_points

# Filters nearby points
def filter_nearby_points(new_pts, existing_pts, min_dist=25):
    tree = cKDTree(existing_pts)  # type: ignore
    dists, _ = tree.query(new_pts, k=1)  # type: ignore
    return new_pts[dists >= min_dist]

# Reads a single-channel depth map from an EXR file
def read_exr(file_path, depth_channel="R", depth_scale=5.0):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    if depth_channel in header['channels']:
        channel_data = exr_file.channel(depth_channel, pt)
        depth = np.frombuffer(channel_data, dtype=np.float32).reshape(size[1], size[0]).copy()

        # Apply depth scale
        depth *= depth_scale

        return depth
    else:
        raise ValueError(f"Channel '{depth_channel}' not found in EXR file.")
    
# Computes depth map from stereo images or uses a ground truth depth map from an EXR file.
def compute_depth(gt_depth_file=None, depth_scale=5.0, filter_depth=True):
    if gt_depth_file is None:
        raise ValueError("Ground truth depth file must be provided when use_ground_truth is True.")
    try:
        depth_map = read_exr(gt_depth_file, depth_channel="R")
        
        if depth_map is None:
            raise FileNotFoundError(f"Ground truth depth file not found: {gt_depth_file}")
        
        depth_map *= depth_scale
        print(f"✅ Loaded ground truth depth map from: {gt_depth_file}")
        
    except Exception as e:
        raise RuntimeError(f"Error reading EXR file: {e}")

    if filter_depth:
        depth_min = np.nanpercentile(depth_map, 0)
        depth_max = np.nanpercentile(depth_map, 95)
        valid_mask = (depth_map >= depth_min) & (depth_map <= depth_max) & ~np.isnan(depth_map)
        filtered_depth_map = np.where(valid_mask, depth_map, np.nan)
    else:
        filtered_depth_map = depth_map

    return filtered_depth_map

# Bilinear interpolation for depth map
def bilinear_interpolate(depth_map, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, depth_map.shape[1] - 1)
    x1 = np.clip(x1, 0, depth_map.shape[1] - 1)
    y0 = np.clip(y0, 0, depth_map.shape[0] - 1)
    y1 = np.clip(y1, 0, depth_map.shape[0] - 1)

    Ia = depth_map[y0, x0]
    Ib = depth_map[y1, x0]
    Ic = depth_map[y0, x1]
    Id = depth_map[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

# Uses Mast3r to track features
def track_features_mast3r(original_image_path, tracked_image_path, valid_pixels, model):
    images = load_images([original_image_path, tracked_image_path], size=512)
    
    # Run inference
    output = inference([tuple(images)], model, 'cuda', batch_size=1, verbose=False)
    # Get keypoints and descriptors
    desc1 = output['pred1']['desc'].squeeze(0).detach()
    desc2 = output['pred2']['desc'].squeeze(0).detach()
    
    matches_im0, matches_im1, status, _ = fast_reciprocal_NNs_SP(desc1, desc2, valid_pixels[:, 0], valid_pixels[:, 1], device='cuda')

    return matches_im0, matches_im1, status
    
# Transforms 3D points from camera coordinates to world coordinates using a specific frame's pose.
def transform_camera_to_world(points_3d_cam, pose_npz_path, frame_idx):
    # Load pose matrix
    data = np.load(pose_npz_path)
    if hasattr(data, 'files'):  # Check if it's an NpzFile
        pose_flat = data[list(data.files)[0]][frame_idx]  # e.g., 'data'
    else:
        pose_flat = data[frame_idx]

    pose_matrix = pose_flat.reshape(4, 4)  # (4, 4)

    # Convert points to homogeneous coordinates
    points_h = np.hstack([points_3d_cam, np.ones((points_3d_cam.shape[0], 1))])  # (N, 4)

    # Apply transformation
    points_world_h = (pose_matrix @ points_h.T).T  # (N, 4)
    return points_world_h[:, :3]  # Drop homogeneous coord

# Transforms 3D points from world coordinates to camera coordinates using a specific frame's pose.
def transform_world_to_camera(points_3d_world, pose_npz_path, frame_idx):
    # Load pose matrix
    data = np.load(pose_npz_path)
    if hasattr(data, 'files'):  # Check if it's an NpzFile
        pose_flat = data[list(data.files)[0]][frame_idx]  # e.g., 'data'
    else:
        pose_flat = data[frame_idx]

    pose_matrix = pose_flat.reshape(4, 4)  # (4, 4)

    # Invert the pose to go from world → camera
    pose_matrix_inv = np.linalg.inv(pose_matrix)

    # Convert points to homogeneous coordinates
    points_h = np.hstack([points_3d_world, np.ones((points_3d_world.shape[0], 1))])  # (N, 4)

    # Apply inverse transformation
    points_cam_h = (pose_matrix_inv @ points_h.T).T  # (N, 4)

    return points_cam_h[:, :3]  # Drop homogeneous coordinate

# Function to save a 3D point cloud to a file
def save_point_cloud(points_3d, filename):
    pcd = o3d.geometry.PointCloud()  # Initialize Open3D PointCloud
    pcd.points = o3d.utility.Vector3dVector(points_3d)  # Add points to the PointCloud
    o3d.io.write_point_cloud(filename, pcd)  # Save the PointCloud to a file
    # print(f"Point cloud saved to {filename}")

# Function to generate a 2D triangular mesh save it in OFF format and process to .npz
def generate_and_process_2d_mesh(valid_pixels):
    # Initialize constrained Delaunay triangulation
    triangulation = cdt.Triangulation(
        cdt.VertexInsertionOrder.AS_PROVIDED,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0
    )
    
    # Convert valid pixels to CDT vertices and perform triangulation
    vertices_2d = [cdt.V2d(x, y) for x, y in valid_pixels]
    triangulation.insert_vertices(vertices_2d)
    triangulation.erase_super_triangle()
    
    # Extract mesh data for .npz format
    mesh_pos = np.array([[v.x, v.y] for v in triangulation.vertices_iter()], dtype=np.float32)
    cells = np.array([[t.vertices[0], t.vertices[1], t.vertices[2]] for t in triangulation.triangles_iter()], dtype=np.int32)
    
    return mesh_pos, cells

# Visualizes 2D points on an image and saves the result
def visualize_2d_points(image, valid_pixels, output_filename):
    # ===== 1. Save scatter overlay =====
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(valid_pixels[:, 0], valid_pixels[:, 1], c='yellow', s=10, label="Tracked Points")
    plt.title("Tracked Points on Frame")
    plt.legend()
    plt.axis("off")
    plt.savefig(output_filename)
    plt.close()

    # ===== 2. Save original image =====
    # Create 'original_image' subfolder in same directory
    output_dir = os.path.dirname(output_filename)
    original_image_dir = os.path.join(output_dir, "original_image")
    os.makedirs(original_image_dir, exist_ok=True)

    # Get filename only (without folder)
    original_name = os.path.basename(output_filename)
    
    # Save the original image (converted to RGB for consistency)
    original_image_path = os.path.join(original_image_dir, original_name)
    cv2.imwrite(original_image_path, image)

# Computes dense optical flow between two frames using NeuFlow
def compute_dense_optical_flow_neuf(flow_path):
    flow = np.load(flow_path)
    print("✅ Loaded flow from ", flow_path)
    return flow

# Computes deformation-only optical flow by subtracting rigid flow from NeuFlowV2 flow.
def compute_dense_deformation_flow_neuf(flow_path, depth_t_minus_1, K, pose_npz_path, frame_idx_t_minus_1, frame_idx_t):
    # Load NeuFlowV2 dense flow (from frame t-1 → t)
    flow = np.load(flow_path).astype(np.float32)  # [H, W, 2]
    H, W = flow.shape[:2]

    # === Step 1: Create pixel grid for frame t-1 ===
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u)
    pix_coords = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T  # [3, H*W]
    z = depth_t_minus_1.flatten()  # [H*W]

    # === Step 2: Back-project pixels to 3D in camera_{t-1} frame ===
    K_inv = np.linalg.inv(K)
    X_cam_t_minus_1 = (K_inv @ (pix_coords * z)).T  # [H*W, 3]

    # === Step 3: Transform 3D points to world space using pose at frame t-1 ===
    X_world = transform_camera_to_world(X_cam_t_minus_1, pose_npz_path, frame_idx_t_minus_1)  # [H*W, 3]

    # === Step 4: Transform world points into camera_t space using the inverse function ===
    X_cam_t = transform_world_to_camera(X_world, pose_npz_path, frame_idx_t)  # [H*W, 3]

    # === Step 5: Project 3D points to 2D in frame t ===
    X_cam_homo = X_cam_t.T  # [3, H*W]
    proj = K @ X_cam_homo   # [3, H*W]
    proj = (proj[:2] / proj[2:]).T  # [H*W, 2]
    rigid_uv = proj.reshape(H, W, 2)

    # === Step 6: Compute rigid flow and subtract it from NeuFlow ===
    coords_t_minus_1 = np.stack([u, v], axis=-1).astype(np.float32)  # [H, W, 2]
    rigid_flow = rigid_uv - coords_t_minus_1  # [H, W, 2]
    deformation_flow = flow - rigid_flow

    return deformation_flow

# Extracts and filters patched flow
def extract_and_filter_patched_flow(image_positions, flow_maps, world_positions, patch_size=6):
    half_patch = patch_size // 2
    frame_keys = sorted(flow_maps.keys())  # e.g., frame_0002 to frame_XXXX

    patched_flow = {}

    for frame_key in frame_keys:
        flow_map = flow_maps[frame_key]  # (H, W, 2)
        H, W, _ = flow_map.shape

        points_2d = image_positions[frame_key]      # (N_i, 2)
        points_3d = world_positions[frame_key]       # (N_i, 3)
        num_points = points_2d.shape[0]

        # Init per-frame patch array
        patched_flow[frame_key] = np.full((num_points, patch_size, patch_size, 2), np.nan, dtype=np.float32)

        for i in range(num_points):
            x, y = points_2d[i]
            if np.isnan(x) or np.isnan(y):
                continue  # Already invalid

            x_int, y_int = int(x), int(y)

            if (
                x_int - half_patch < 0 or x_int + half_patch >= W or
                y_int - half_patch < 0 or y_int + half_patch >= H
            ):
                # Mark point as invalid in all maps
                image_positions[frame_key][i] = np.nan
                world_positions[frame_key][i] = np.nan
                patched_flow[frame_key][i] = np.full((patch_size, patch_size, 2), np.nan, dtype=np.float32)
            else:
                patched_flow[frame_key][i] = flow_map[
                    y_int - half_patch : y_int + half_patch,
                    x_int - half_patch : x_int + half_patch,
                    :
                ]

    return image_positions, world_positions, patched_flow

# Pads all per-frame arrays in a .npz file to the same number of points (N=max), using NaN for missing entries.
def pad_npz_dict(data_dict):
    keys = list(data_dict.keys())
    max_points = max(data_dict[key].shape[0] for key in keys)

    padded_data = {}
    for key in keys:
        arr = data_dict[key]
        N = arr.shape[0]

        if N == max_points:
            padded_data[key] = arr
        else:
            pad_shape = (max_points - N,) + arr.shape[1:]
            pad_block = np.full(pad_shape, np.nan, dtype=arr.dtype)
            padded_arr = np.concatenate([arr, pad_block], axis=0)
            padded_data[key] = padded_arr.astype(np.float32)

    return padded_data

# Removes NaN points from world positions
def remove_nan_points(world_positions):
    # Load the frames
    frame_keys = sorted(world_positions.keys())
    all_frames = [world_positions[key] for key in frame_keys]
    all_points = np.stack(all_frames, axis=0)  # Shape: (num_frames, num_points, 3)

    # Find points with NaN values across any frame
    valid_points_mask = ~np.any(np.isnan(all_points), axis=(0, 2))  # Shape: (num_points,)
    
    # Filter world positions
    filtered_world_positions = {key: world_positions[key][valid_points_mask] for key in frame_keys}
    
    return filtered_world_positions, valid_points_mask

# Tracks points across frames and generates 3D world positions.
def track_points_and_generate_world_positions(
    start_frame_index, end_frame_index, folder_path, prefix, valid_pixels, K,
    output_folder, points_3d_0_world, points_3d_1_world, points_2d_0, points_2d_1, 
    flow_decouple=True, model=None, patch_size=6, train_eval="train", extra_grid_size=12):
    
    # Load sorted image file lists
    images = sorted([img for img in os.listdir(folder_path) if img.startswith(f"{prefix}_0")])
    depth_files = sorted([f for f in os.listdir(folder_path) if f.startswith(f"{prefix}_depth")])
    flow_paths = sorted([f for f in os.listdir(os.path.join(folder_path, "flow_map")) if f.startswith(f"{prefix}_0")])
    
    # Initialize tracking data
    tracked_points = valid_pixels.copy()
    all_tracked_points = []  # Stores valid_tracked_points for all frames
    output_ply_folder = os.path.join(output_folder, "output_ply")
    output_png_folder = os.path.join(output_folder, "output_png")
    os.makedirs(output_ply_folder, exist_ok=True)
    os.makedirs(output_png_folder, exist_ok=True)
    min_alive_points = 175

    # === Initialize World Positions (3D) and Image Positions (2D) ===
    world_positions = {
        "frame_0000": points_3d_0_world,
        "frame_0001": points_3d_1_world
    }

    image_positions = {
        "frame_0000": points_2d_0,
        "frame_0001": points_2d_1
    }
    
    flow_maps = {} 

    # Start Tracking from frame_2 because 0000 & 0001 are stored
    for frame_index in range(start_frame_index, end_frame_index + 1):
        I_t_path = os.path.join(folder_path, images[frame_index])
        I_t = cv2.imread(I_t_path)

        # Compute Dense Optical Flow (from frame_{t-1} → frame_t) using TV-L1
        I_prev_path = os.path.join(folder_path, images[frame_index - 1])
        I_prev = cv2.imread(I_prev_path)
        
        if flow_decouple:
            flow_map = compute_dense_deformation_flow_neuf(
                flow_path=os.path.join(folder_path, "flow_map", flow_paths[frame_index - 1]),
                depth_t_minus_1=compute_depth(os.path.join(folder_path, depth_files[frame_index - 1]), filter_depth=False),
                K=K,
                pose_npz_path=f"{folder_path}{prefix}_pose_matrices.npz",
                frame_idx_t_minus_1=frame_index - 1,
                frame_idx_t=frame_index
            )
        else:
            flow_map = compute_dense_optical_flow_neuf(os.path.join(folder_path, "flow_map", flow_paths[frame_index-1]))
            
        flow_maps[f"frame_{frame_index:04d}"] = flow_map.astype(np.float32)

        # Track points from previous frame
        if frame_index > start_frame_index:

            print("Using CoTracker for feature tracking.")

            # Select only alive points (not previously lost)
            tracked_subset = tracked_points[alive_point_mask]

            # Track only valid ones
            _, tracked_subset, status = track_features_mast3r(I_prev_path, I_t_path, tracked_subset, model)
            tracked_subset = tracked_subset.reshape(-1, 2)

            # Filter based on CoTracker status and image bounds
            valid_mask_subset = (
                (status == 1) &
                (tracked_subset[:, 0] >= 0) & (tracked_subset[:, 0] < I_t.shape[1]) &
                (tracked_subset[:, 1] >= 0) & (tracked_subset[:, 1] < I_t.shape[0])
            )

            # Create full-size placeholder and insert valid tracked points
            new_tracked_points = np.full_like(tracked_points, np.nan)
            new_tracked_points[alive_point_mask] = np.where(
                valid_mask_subset[:, None], tracked_subset, np.nan
            )

            # Update the masks and points
            tracked_points = new_tracked_points
            valid_tracked_points = tracked_points.copy()
            alive_point_mask = ~np.isnan(tracked_points).any(axis=1)

        else:
            # First tracked frame
            valid_tracked_points = tracked_points.copy()
            alive_point_mask = np.ones(tracked_points.shape[0], dtype=bool)

        if np.sum(alive_point_mask) < min_alive_points and train_eval == "train":
            candidate_points = create_grid_points(I_t.shape, grid_size=extra_grid_size, use_jitter=True)
            existing_points = tracked_points[alive_point_mask]
            new_points = filter_nearby_points(candidate_points, existing_points, min_dist=25)
            if len(new_points) > 0:
                print(f"Injecting {len(new_points)} new points at frame {frame_index}")
                tracked_points = np.vstack([tracked_points, new_points])
                valid_tracked_points = tracked_points.copy()
                alive_point_mask = np.concatenate([alive_point_mask, np.ones(len(new_points), dtype=bool)])
            
                
        # Store tracked points for the current frame
        all_tracked_points.append(valid_tracked_points)

        # === Compute Depth Map for the Current Frame ===
        depth_file = os.path.join(folder_path, depth_files[frame_index])
        filtered_depth_map = compute_depth(gt_depth_file=depth_file)

        # === Compute 3D Points ===
        points_3d_cam = np.full((len(valid_tracked_points), 3), np.nan)

        valid_mask = ~np.isnan(valid_tracked_points[:, 0]) & ~np.isnan(valid_tracked_points[:, 1])
        Z = np.full(len(valid_tracked_points), np.nan)
        valid_indices = valid_mask & (
            (valid_tracked_points[:, 0] < filtered_depth_map.shape[1]) &
            (valid_tracked_points[:, 1] < filtered_depth_map.shape[0])
        )
        Z[valid_indices] = bilinear_interpolate(filtered_depth_map, valid_tracked_points[valid_indices, 0], valid_tracked_points[valid_indices, 1])

        # Convert to 3D world positions
        depth_valid_mask = ~np.isnan(Z)
        points_3d_cam[depth_valid_mask] = np.column_stack((
            (valid_tracked_points[depth_valid_mask, 0] - K[0, 2]) * Z[depth_valid_mask] / K[0, 0],
            (valid_tracked_points[depth_valid_mask, 1] - K[1, 2]) * Z[depth_valid_mask] / K[1, 1],
            Z[depth_valid_mask]
        ))
        
        # Transform camera to world
        points_3d_world_filtered = transform_camera_to_world(points_3d_cam, pose_npz_path = f"{folder_path}{prefix}_pose_matrices.npz", frame_idx = frame_index)
        valid_tracked_points[~depth_valid_mask] = np.nan

        # Store 3D world positions and 2D image positions
        world_positions[f"frame_{frame_index:04d}"] = points_3d_world_filtered.astype(np.float32)
        image_positions[f"frame_{frame_index:04d}"] = valid_tracked_points.astype(np.float32)

    # === Extract and Filter Patched Flow ===
    image_positions, world_positions, patched_flow = extract_and_filter_patched_flow(image_positions=image_positions, flow_maps=flow_maps, world_positions=world_positions, patch_size=patch_size)

    # === Save 3D & 2D Visualizations ===
    if train_eval == "train":
        print("Skipping 3D & 2D Visualizations for evaluation.")
        # === Save .npz world positions (with padding)
        world_positions= pad_npz_dict(world_positions)
        # world_positions_path = os.path.join(output_folder, "world_positions.npz")
        # np.savez(world_positions_path, **world_positions)

        # === Save .npz image positions (with padding)
        image_positions = pad_npz_dict(image_positions)
        # image_positions_path = os.path.join(output_folder, "image_positions.npz")
        # np.savez(image_positions_path, **image_positions)

        # === Save .npz patched flow maps (with padding)
        patched_flow = pad_npz_dict(patched_flow)
        # patched_flow_path = os.path.join(output_folder, "patched_flow.npz")
        # np.savez(patched_flow_path, **patched_flow)
    else:
        world_positions, valid_points_mask = remove_nan_points(world_positions)
        image_positions = {key: image_positions[key][valid_points_mask] for key in image_positions.keys()}
        patched_flow = {key: patched_flow[key][valid_points_mask] for key in patched_flow.keys()}
        
        print(f"Saving 3D points and visualizations...")
        for frame_index in range(start_frame_index, end_frame_index + 1):
            frame_key = f"frame_{frame_index:04d}"
            frame_name = f"frame_{frame_index - 1:04d}"

            points_3d_world = world_positions[frame_key]
            points_2d_image = image_positions[frame_key]

            # Create a valid mask: keep only points where both 3D and 2D are valid
            valid_mask = (
                ~np.isnan(points_3d_world).any(axis=1) &
                ~np.isnan(points_2d_image).any(axis=1)
            )
            points_3d_world_filtered = points_3d_world[valid_mask]
            valid_tracked_points_filtered = points_2d_image[valid_mask]

            I_t = cv2.imread(os.path.join(folder_path, images[frame_index]))
            save_point_cloud(points_3d_world_filtered, os.path.join(output_ply_folder, f"{frame_name}.ply"))
            visualize_2d_points(I_t, valid_tracked_points_filtered, os.path.join(output_png_folder, f"{frame_name}.png"))

    
    # === Save .npz dense optical flow maps (no padding needed)
    flow_maps_path = os.path.join(output_folder, "flow_maps.npz")
    np.savez(flow_maps_path, **flow_maps)
    
    return world_positions, image_positions, flow_maps, patched_flow

# Creates and saves dataset with patched flow, including average target patched flow
def create_and_save_dataset_train(world_positions, image_positions, patched_flow, output_file, tensor_output_folder):
    # === Sort frames ===
    sorted_frames = sorted(world_positions.keys())  # Frames 0000 onwards

    # === Dictionary format dataset ===
    dataset = {}
    for i in range(2, len(sorted_frames) - 1):  # Start from frame_0002
        current_frame_key = sorted_frames[i]
        prev_frame_key = sorted_frames[i - 1]
        prev_prev_frame_key = sorted_frames[i - 2]
        target_frame_key = sorted_frames[i + 1]

        world_pos = world_positions[current_frame_key].astype(np.float32)
        prev_world_pos = world_positions[prev_frame_key].astype(np.float32)
        prev_prev_world_pos = world_positions[prev_prev_frame_key].astype(np.float32)
        target_world_pos = world_positions[target_frame_key].astype(np.float32)
        
        # === Build a single valid mask ===
        valid_mask = (
            ~np.isnan(world_pos).any(axis=1) &
            ~np.isnan(prev_world_pos).any(axis=1) &
            ~np.isnan(prev_prev_world_pos).any(axis=1) &
            ~np.isnan(target_world_pos).any(axis=1)
        )
        
        # === Filter world positions
        world_pos           = world_pos[valid_mask]
        prev_world_pos      = prev_world_pos[valid_mask]
        prev_prev_world_pos = prev_prev_world_pos[valid_mask]
        target_world_pos    = target_world_pos[valid_mask]


        # === Filter image positions ===
        image_pos           = image_positions[current_frame_key][valid_mask].astype(np.float32)
        prev_image_pos      = image_positions[prev_frame_key][valid_mask].astype(np.float32)
        prev_prev_image_pos = image_positions[prev_prev_frame_key][valid_mask].astype(np.float32)
        target_image_pos    = image_positions[target_frame_key][valid_mask].astype(np.float32)

        # === Filter patched flow ===
        patched_flow_current     = patched_flow[current_frame_key][valid_mask].astype(np.float32)  # (N, 6, 6, 2)
        target_patched_flow      = patched_flow[target_frame_key][valid_mask].astype(np.float32)   # (N, 6, 6, 2)
        avg_target_patched_flow  = np.mean(target_patched_flow, axis=(1, 2))  # → shape: (N, 2)

        # === Generate mesh from filtered 2D image positions
        mesh_pos, cells = generate_and_process_2d_mesh(image_pos)
        node_type = np.zeros((mesh_pos.shape[0], 1), dtype=np.int32)

        # === Construct the dataset for the current frame ===
        dataset[current_frame_key] = {
            "cells": cells.astype(np.int32),
            "mesh_pos": mesh_pos.astype(np.float32),
            "node_type": node_type.astype(np.int32),
            "world_pos": world_pos,
            "prev|world_pos": prev_world_pos,
            "prev_prev|world_pos": prev_prev_world_pos,
            "target|world_pos": target_world_pos,
            "image_pos": image_pos,
            "prev|image_pos": prev_image_pos,
            "prev_prev|image_pos": prev_prev_image_pos,
            "target|image_pos": target_image_pos,
            "patched_flow": patched_flow_current,
            "target|patched_flow": target_patched_flow,
            "avg_target|patched_flow": avg_target_patched_flow,
        }
    # === Save dictionary format dataset ===
    np.savez(output_file, **{f"{frame_key}/{key}": value for frame_key, frame_data in dataset.items() for key, value in frame_data.items()})
    print(f"Entire dictionary format dataset saved to {output_file}")

    # === Save each frame as an individual .npz file in the tensor_output_folder ===
    os.makedirs(tensor_output_folder, exist_ok=True)

    for i in range(2, len(sorted_frames) - 1):
        frame_key = sorted_frames[i]
        frame_data = dataset[frame_key]

        shifted_index = i - 1  # e.g., 2 → 1 (frame_0001.npz)
        output_path = os.path.join(tensor_output_folder, f"frame_{shifted_index:04d}.npz")

        def expand(x):
            return np.expand_dims(x, axis=0)  # Add time dimension [1, ...]

        np.savez(output_path, **{
            "cells": expand(frame_data["cells"].astype(np.int32)),  # Already shape [M, 3], keep as is
            "mesh_pos": expand(frame_data["mesh_pos"].astype(np.float32)),
            "node_type": expand(frame_data["node_type"].astype(np.int32)),
            "world_pos": expand(frame_data["world_pos"].astype(np.float32)),
            "prev|world_pos": expand(frame_data["prev|world_pos"].astype(np.float32)),
            "prev_prev|world_pos": expand(frame_data["prev_prev|world_pos"].astype(np.float32)),
            "target|world_pos": expand(frame_data["target|world_pos"].astype(np.float32)),
            "image_pos": expand(frame_data["image_pos"].astype(np.float32)),
            "prev|image_pos": expand(frame_data["prev|image_pos"].astype(np.float32)),
            "prev_prev|image_pos": expand(frame_data["prev_prev|image_pos"].astype(np.float32)),
            "target|image_pos": expand(frame_data["target|image_pos"].astype(np.float32)),
            "patched_flow": expand(frame_data["patched_flow"].astype(np.float32)),
            "target|patched_flow": expand(frame_data["target|patched_flow"].astype(np.float32)),
            "avg_target|patched_flow": expand(frame_data["avg_target|patched_flow"].astype(np.float32)),
        })

    print(f"✅ Saved dataset to {tensor_output_folder}")

# Creates and saves dataset with patched flow, including average target patched flow
def create_and_save_dataset_eval(world_positions, image_positions, patched_flow, output_file, tensor_output_file):
    sorted_frames = sorted(world_positions.keys())  # e.g., frame_0000, frame_0001, ...

    dataset = {}
    for i in range(2, len(sorted_frames) - 1):
        current_frame_key = sorted_frames[i]
        prev_frame_key = sorted_frames[i - 1]
        prev_prev_frame_key = sorted_frames[i - 2]
        target_frame_key = sorted_frames[i + 1]

        world_pos = world_positions[current_frame_key].astype(np.float32)
        prev_world_pos = world_positions[prev_frame_key].astype(np.float32)
        prev_prev_world_pos = world_positions[prev_prev_frame_key].astype(np.float32)
        target_world_pos = world_positions[target_frame_key].astype(np.float32)

        image_pos = image_positions[current_frame_key].astype(np.float32)
        prev_image_pos = image_positions[prev_frame_key].astype(np.float32)
        prev_prev_image_pos = image_positions[prev_prev_frame_key].astype(np.float32)
        target_image_pos = image_positions[target_frame_key].astype(np.float32)

        patched_flow_current = patched_flow[current_frame_key].astype(np.float32)
        target_patched_flow = patched_flow[target_frame_key].astype(np.float32)
        avg_target_patched_flow = np.mean(target_patched_flow, axis=(1, 2))

        
        if i == 2:
            valid_pixel_coords = image_positions[current_frame_key]
            mesh_pos, cells = generate_and_process_2d_mesh(valid_pixel_coords)
            node_type = np.zeros((mesh_pos.shape[0], 1), dtype=np.int32)

        dataset[current_frame_key] = {
            "cells": cells,
            "mesh_pos": mesh_pos,
            "node_type": node_type,
            "world_pos": world_pos,
            "prev|world_pos": prev_world_pos,
            "prev_prev|world_pos": prev_prev_world_pos,
            "target|world_pos": target_world_pos,
            "image_pos": image_pos,
            "prev|image_pos": prev_image_pos,
            "prev_prev|image_pos": prev_prev_image_pos,
            "target|image_pos": target_image_pos,
            "patched_flow": patched_flow_current,
            "target|patched_flow": target_patched_flow,
            "avg_target|patched_flow": avg_target_patched_flow,
        }

    # Save dictionary-style dataset
    np.savez(output_file, **{f"{frame_key}/{key}": value
              for frame_key, frame_data in dataset.items()
              for key, value in frame_data.items()})
    print(f"Entire dictionary format dataset saved to {output_file}")

    # Stack for .npz tensor-style saving
    stacked_tensors = {
        key: np.stack([dataset[sorted_frames[i]][key] for i in range(2, len(sorted_frames) - 1)], axis=0)
        for key in dataset[sorted_frames[2]].keys()
    }

    np.savez(tensor_output_file, **stacked_tensors)
    print(f"Stacked tensors saved to {tensor_output_file}")

# Generates a SLAM dataset by performing feature tracking, world position generation, and dataset saving.
def dataset_generation(traj_num, start_frame_index, end_frame_index, grid_size, patch_size, train_eval = "train", extra_grid_size=12, base_source="./mono_datasets/"):
    # Define paths, camera intrinsics, and baseline distance
    data_folder = f'{base_source}monocular_deformable_{traj_num}/'
    prefix = detect_prefix(data_folder)  # Auto-detect prefix
    print(f"Using Prefix: {prefix}")

    K = np.array([[248.12, 0, 256], [0, 248.12, 192], [0, 0, 1]])  # Camera intrinsic matrix
    
    output_folder = f"./simulator_data/trajectory_{traj_num}/trj_{traj_num}_ts_{end_frame_index-1}"
    os.makedirs(output_folder, exist_ok=True)

    # === Load Image for Three Frames ===
    I_2_path = f"{data_folder}{prefix}_00002.png"
    I_1_path = f"{data_folder}{prefix}_00001.png"
    I_0_path = f"{data_folder}{prefix}_00000.png"
    
    # I_2 = cv2.imread(I_2_path)
    I_1 = cv2.imread(I_1_path)
    I_0 = cv2.imread(I_0_path)

    # Initialize grid points and compute valid pixels from depth map
    grid_points = create_grid_points(I_1.shape, grid_size=grid_size, margin_ratio=0.1)
    
    filtered_depth_map_2 = compute_depth(gt_depth_file=f"{data_folder}{prefix}_depth00002.exr")
    valid_pixels_2 = grid_points[~np.isnan(filtered_depth_map_2[grid_points[:, 1], grid_points[:, 0]])]

    # === Feature Tracking: I_2 → I_1 → I_0 ===
    print("Using Mast3r for feature tracking.")
    # model_name = "./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to('cuda')

    # Track from I_2 → I_1
    original_points, tracked_points_1, status_1 = track_features_mast3r(I_2_path, I_1_path, valid_pixels_2, model)
    # Track from I_1 → I_0
    tracked_points_1, tracked_points_0, status_0 = track_features_mast3r(I_1_path, I_0_path, tracked_points_1, model)


    # === Filter Valid Tracked Points Across Frames (Right After Tracking) ===
    valid_mask_1 = (
        (status_1 == 1) &
        (tracked_points_1[:, 0] >= 0) & (tracked_points_1[:, 0] < I_1.shape[1]) &
        (tracked_points_1[:, 1] >= 0) & (tracked_points_1[:, 1] < I_1.shape[0])
    )
    
    valid_mask_0 = (
        (status_0 == 1) &
        (tracked_points_0[:, 0] >= 0) & (tracked_points_0[:, 0] < I_0.shape[1]) &
        (tracked_points_0[:, 1] >= 0) & (tracked_points_0[:, 1] < I_0.shape[0])
    )

    valid_original = original_points[valid_mask_1 & valid_mask_0].reshape(-1, 2)
    valid_tracked_1 = tracked_points_1[valid_mask_1 & valid_mask_0].reshape(-1, 2)
    valid_tracked_0 = tracked_points_0[valid_mask_1 & valid_mask_0].reshape(-1, 2)

    # === Compute Depth Maps for I_1 and I_0 (After Tracking and Filtering) ===
    filtered_depth_map_1 = compute_depth(gt_depth_file=f"{data_folder}{prefix}_depth00001.exr")
    filtered_depth_map_0 = compute_depth(gt_depth_file=f"{data_folder}{prefix}_depth00000.exr")
    
    # === Filter Tracked Points Again Based on Valid Depths ===
    valid_depth_1 = ~np.isnan(filtered_depth_map_1[valid_tracked_1[:, 1].astype(int), valid_tracked_1[:, 0].astype(int)])
    valid_depth_0 = ~np.isnan(filtered_depth_map_0[valid_tracked_0[:, 1].astype(int), valid_tracked_0[:, 0].astype(int)])
    
    valid_mask = valid_depth_1 & valid_depth_0
    valid_original = valid_original[valid_mask]
    valid_tracked_1 = valid_tracked_1[valid_mask]
    valid_tracked_0 = valid_tracked_0[valid_mask]
    
    # === Compute 3D Points for I_1 and I_0 ===
    Z_1 = bilinear_interpolate(filtered_depth_map_1, valid_tracked_1[:, 0], valid_tracked_1[:, 1])
    Z_0 = bilinear_interpolate(filtered_depth_map_0, valid_tracked_0[:, 0], valid_tracked_0[:, 1])
    
    points_3d_1_cam = np.column_stack(((valid_tracked_1[:, 0] - K[0, 2]) * Z_1 / K[0, 0], (valid_tracked_1[:, 1] - K[1, 2]) * Z_1 / K[1, 1], Z_1))
    points_3d_0_cam = np.column_stack(((valid_tracked_0[:, 0] - K[0, 2]) * Z_0 / K[0, 0], (valid_tracked_0[:, 1] - K[1, 2]) * Z_0 / K[1, 1], Z_0))
    
    # Transform 3D points from camera coordinates to world coordinates
    points_3d_1_world = transform_camera_to_world(points_3d_1_cam, pose_npz_path = f"{data_folder}{prefix}_pose_matrices.npz", frame_idx = 1)
    points_3d_0_world = transform_camera_to_world(points_3d_0_cam, pose_npz_path = f"{data_folder}{prefix}_pose_matrices.npz", frame_idx = 0)
    
    # Perform tracking and generate world positions for multiple frames
    world_positions, image_positions, flow_maps, patched_flow = track_points_and_generate_world_positions(
        start_frame_index=start_frame_index,
        end_frame_index=end_frame_index,
        folder_path=data_folder,
        prefix=prefix,
        valid_pixels=valid_original,
        K=K,
        output_folder=output_folder,
        points_3d_0_world=points_3d_0_world,
        points_3d_1_world=points_3d_1_world,
        points_2d_0=valid_tracked_0,
        points_2d_1=valid_tracked_1,
        model=model,
        patch_size=patch_size,
        train_eval=train_eval,
        extra_grid_size=extra_grid_size
    )

    # Create and save SLAM dataset in one step
    if train_eval == "train":
        create_and_save_dataset_train(world_positions, image_positions, patched_flow, os.path.join(output_folder, "dic_dataset.npz"), os.path.join(output_folder, f'trj_{traj_num}_tensor'))
    else:
        create_and_save_dataset_eval(world_positions, image_positions, patched_flow, os.path.join(output_folder, "dic_dataset_eval.npz"), os.path.join(output_folder, f'trj_{traj_num}_eval_tensor.npz'))

    print("Dataset creation and processing completed.")
    
    return output_folder

# Copy tensor folder to central destination
def copy_tensor_folder(traj_num_str, output_folder):
    tensor_folder_name = f"trj_{traj_num_str}_tensor"
    tensor_src = os.path.join(output_folder, tensor_folder_name)

    # Extract the leading digit to define the group folder
    group_id = traj_num_str[0]  # e.g., "1" from "1001"
    group_folder = os.path.join("./input/simulator", f"trj_{group_id}")
    tensor_dst = os.path.join(group_folder, tensor_folder_name)

    # Ensure destination directory exists
    os.makedirs(group_folder, exist_ok=True)

    # Replace if already exists
    if os.path.exists(tensor_dst):
        shutil.rmtree(tensor_dst)
    shutil.copytree(tensor_src, tensor_dst)

    print(f"✅ Copied {tensor_src} → {tensor_dst}")


def detect_traj_nums(base_source):
    import os, re
    root = base_source.rstrip('/')
    pattern = re.compile(r'^monocular_deformable_(\d{4,5})$')
    traj_nums = []
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        m = pattern.match(name)
        if m:
            traj_nums.append(int(m.group(1)))
    return sorted(traj_nums)


# Automatically generate datasets for all trajectories and grid sizes
def auto_generate_datasets(base_source="./mono_datasets/", start_frame_index=2, end_frame_index=22,
                           patch_size=6, train_eval="eval", extra_grid_size=12):
    traj_list = detect_traj_nums(base_source)
    for traj_num in traj_list:
        traj_num_str = str(traj_num)
        grid_size = extra_grid_size
        print(f"Generating dataset for Trajectory {traj_num}, Grid Size {grid_size}...")
        output_folder = dataset_generation(traj_num_str, start_frame_index, end_frame_index, grid_size, patch_size,
                                           train_eval=train_eval, extra_grid_size=extra_grid_size, base_source=base_source)
        if train_eval == "train":
            copy_tensor_folder(traj_num_str, output_folder)

# if __name__ == "__main__":
#     base_source = "./mono_datasets/"
#     start_frame_index = 2
#     end_frame_index = 22
#     patch_size = 6
#     train_eval = "train"
#     extra_grid_size = 20
#     auto_generate_datasets(base_source=base_source,
#                            start_frame_index=start_frame_index,
#                            end_frame_index=end_frame_index,
#                            patch_size=patch_size,
#                            train_eval=train_eval,
#                            extra_grid_size=extra_grid_size)
