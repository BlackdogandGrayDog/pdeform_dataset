import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os

def detect_prefix(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        match = re.match(r"^([A-Z]+)_\d{5}\.(png|jpg)$", file)
        if match:
            return match.group(1)
    raise ValueError("Could not detect a valid prefix from image filenames in the folder!")

# Auto-detects available frame indices and image format
def detect_frame_range_and_format(image_dir, prefix):
    image_dir = Path(image_dir)
    png_files = sorted(image_dir.glob(f"{prefix}_*.png"))
    jpg_files = sorted(image_dir.glob(f"{prefix}_*.jpg"))

    if png_files and jpg_files:
        raise ValueError("Directory contains both PNG and JPG files. Only one format should be used.")

    image_files = png_files if png_files else jpg_files
    if not image_files:
        raise FileNotFoundError("No image files found with the given prefix.")

    frame_pattern = re.compile(rf"{prefix}_(\d+)\.(png|jpg)")
    frame_indices = []
    for f in image_files:
        m = frame_pattern.match(f.name)
        if m:
            frame_indices.append(int(m.group(1)))

    if not frame_indices:
        raise ValueError("No valid frame indices found in image filenames.")

    return sorted(set(frame_indices)), image_files[0].suffix.lower()

# === Utilities ===
def parse_quaternion_file(path):
    quats = {}
    pattern = re.compile(
        r"Frame (\d+)\s+Rotation: X=([-0-9.eE]+), Y=([-0-9.eE]+), Z=([-0-9.eE]+), W=([-0-9.eE]+)"
    )
    with open(path, "r") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                frame = int(m.group(1))
                x, y, z, w = map(float, m.groups()[1:])
                quats[frame] = [x, y, z, w]
    return quats


def parse_position_file(path):
    poses = {}
    pattern = re.compile(
        r"Frame (\d+)\s+Position: X=([-0-9.eE]+), Y=([-0-9.eE]+), Z=([-0-9.eE]+)"
    )
    with open(path, "r") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                frame = int(m.group(1))
                x, y, z = map(float, m.groups()[1:])
                poses[frame] = [x, y, z]
    return poses


def get_RT_c2w_unity_fixed(q_raw, t_raw):
    x, y, z, w = q_raw
    quat_fixed = [-x, y, -z, w]  # Unity to OpenCV fix
    R_c2w = R.from_quat(quat_fixed).as_matrix()
    tx, ty, tz = t_raw
    T_c2w = np.array([tx, -ty, tz]).reshape(3, 1) * 10.0   # Flip Y and convert cm → mm
    return np.hstack([R_c2w, T_c2w])  # 3x4


# === Main conversion function ===
def pose_matrices_generation(image_dir):
    prefix = detect_prefix(image_dir)
    print(f"🟢 Detected image prefix: {prefix}")

    frame_indices, ext = detect_frame_range_and_format(image_dir, prefix)
    print(f"🟢 Using frame range: {min(frame_indices)} to {max(frame_indices)} with extension: {ext[1:].upper()}")

    quat_path = f"{image_dir}{prefix}_Camera Quaternion Rotation Data.txt"
    pos_path = f"{image_dir}{prefix}_Camera Position Data.txt"
    output_npz_path = f"{image_dir}{prefix}_pose_matrices.npz" 

    quats = parse_quaternion_file(quat_path)
    positions = parse_position_file(pos_path)

    all_frames = sorted(set(quats.keys()) & set(positions.keys()))
    frame_range = [f for f in frame_indices if f in all_frames]
    if not frame_range:
        raise ValueError("No valid frames found in both image files and pose files.")

    RT0 = get_RT_c2w_unity_fixed(quats[frame_range[0]], positions[frame_range[0]])
    RT0_h = np.vstack([RT0, [0, 0, 0, 1]])
    RT0_inv = np.linalg.inv(RT0_h)

    all_pose_matrices = []

    for frame in frame_range:
        RT = get_RT_c2w_unity_fixed(quats[frame], positions[frame])
        RT_h = np.vstack([RT, [0, 0, 0, 1]])
        RT_canon = RT0_inv @ RT_h
        flat_pose = RT_canon.flatten()
        all_pose_matrices.append(flat_pose)

    all_pose_matrices = np.stack(all_pose_matrices)
    np.savez(output_npz_path, data=all_pose_matrices)  # save with named key
    print(f"✅ Canonical flattened pose matrices saved to: {output_npz_path}")

# # === Run ===
# if __name__ == "__main__":
#     image_dir = "./monocular_datasets/monocular_deformable_4/"
#     pose_matrices_generation(image_dir)
