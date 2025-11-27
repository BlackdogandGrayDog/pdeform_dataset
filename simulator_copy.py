import os
import shutil
import re
from simulator_pose import pose_matrices_generation

def detect_prefix(folder_path):
    """Automatically detects the prefix of monocular image files."""
    files = os.listdir(folder_path)
    for file in files:
        match = re.match(r"^([A-Z]+)_\d{5}\.png$", file)  # e.g., USH_00001.png
        if match:
            return match.group(1)
    raise ValueError("❌ Could not detect a valid prefix from image filenames in the folder!")

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sorted_files(folder, keyword=""):
    return sorted([f for f in os.listdir(folder) if keyword in f and os.path.isfile(os.path.join(folder, f))])

def extract_number(f):
    numbers = re.findall(r"\d+", f)
    return int(numbers[0]) if numbers else -1

def copy_files_keep_name(src_folder, dst_folder, selected_files):
    ensure_folder(dst_folder)
    for filename in selected_files:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.copy(src_path, dst_path)
    print(f"✅ Copied {len(selected_files)} files from {src_folder} to {dst_folder}")

def rename_depth_png(folder):
    prefix = detect_prefix(folder)

    def get_num(name):
        return int(re.search(r"\d+", name).group())

    depth_files = sorted([f for f in os.listdir(folder) if f.endswith(".exr")], key=get_num)
    png_files = sorted([f for f in os.listdir(folder) if f.endswith(".png")], key=get_num)

    for idx, old in enumerate(depth_files):  # Starts from 0
        new = f"{prefix}_depth{idx:05d}.exr"
        os.rename(os.path.join(folder, old), os.path.join(folder, new))
        print(f"🔄 Renamed {old} ➔ {new}")

    for idx, old in enumerate(png_files):  # Starts from 0
        new = f"{prefix}_{idx:05d}.png"
        os.rename(os.path.join(folder, old), os.path.join(folder, new))
        print(f"🔄 Renamed {old} ➔ {new}")

def rename_flow(folder):
    parent_folder = os.path.dirname(folder)
    prefix = detect_prefix(parent_folder)

    flow_files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

    for idx, old in enumerate(flow_files):  # Starts from 0
        new = f"{prefix}_{idx:05d}_flow.npy"
        os.rename(os.path.join(folder, old), os.path.join(folder, new))
        print(f"🔄 Renamed {old} ➔ {new}")


def delete_first_flow(folder):
    """Delete the flow file 00001 after renaming, using detected prefix."""
    parent_folder = os.path.dirname(folder)
    prefix = detect_prefix(parent_folder)

    first_flow = os.path.join(folder, f"{prefix}_00000_flow.npy")
    if os.path.exists(first_flow):
        os.remove(first_flow)
        print(f"🗑️ Deleted {first_flow}")
    else:
        print(f"⚠️ Warning: {first_flow} not found to delete.")
        
def copy_camera_metadata_files(source_folder, dest_folder):
    """Detect prefix and copy camera position + quaternion files into the destination."""
    prefix = detect_prefix(source_folder)

    quat_path = os.path.join(source_folder, f"{prefix}_Camera Quaternion Rotation Data.txt")
    pos_path = os.path.join(source_folder, f"{prefix}_Camera Position Data.txt")

    if os.path.exists(quat_path) and os.path.exists(pos_path):
        shutil.copy(quat_path, dest_folder)
        shutil.copy(pos_path, dest_folder)
        print(f"📸 Copied camera metadata files to {dest_folder}")
    else:
        print(f"⚠️ Warning: Camera metadata files not found in {source_folder}")

def process_dataset_copy_shift_minus_one_and_rename(source_folder, dest_folder, frame_start, frame_end):
    print("🚀 Starting dataset copy (flow shift -1) and full renaming...")

    flow_src = os.path.join(source_folder, "flow_map")
    flow_dst = os.path.join(dest_folder, "flow_map")

    ensure_folder(dest_folder)
    ensure_folder(flow_dst)

    # Gather files
    all_depth_files = sorted_files(source_folder, "depth")
    all_png_files = [f for f in os.listdir(source_folder) if f.endswith(".png")]
    all_flow_files = sorted_files(flow_src, "flow")  # FLOW inside flow_map

    all_depth_files.sort(key=extract_number)
    all_png_files.sort(key=extract_number)
    all_flow_files.sort(key=extract_number)

    # Select slices
    selected_depth = all_depth_files[frame_start:frame_end+1]
    selected_png = all_png_files[frame_start:frame_end+1]

    if frame_start == 0:
        # If starting from frame 0, we can't have flow for the first frame (no previous frame)
        selected_flow = all_flow_files[frame_start:frame_end]  # flows from 1 to end
        should_delete_first_flow = False
    else:
        # Shifted flows: flow from T-1 to T is associated with T
        selected_flow = all_flow_files[frame_start-1:frame_end]
        should_delete_first_flow = True

    # Copy files
    copy_files_keep_name(source_folder, dest_folder, selected_depth)
    copy_files_keep_name(source_folder, dest_folder, selected_png)
    copy_files_keep_name(flow_src, flow_dst, selected_flow)
    
    # Copy camera metadata files
    copy_camera_metadata_files(source_folder, dest_folder)
    
    # Generate pose matrices
    pose_matrices_generation(dest_folder)

    # Rename depth and png files, and delete first flow only if required
    if should_delete_first_flow:
        rename_depth_png(dest_folder)
        rename_flow(flow_dst)
        delete_first_flow(flow_dst)

    # Create info.txt
    info_path = os.path.join(dest_folder, "info.txt")
    with open(info_path, "w") as f:
        source_folder_name = os.path.basename(os.path.normpath(source_folder))
        f.write(f"{source_folder_name} {frame_start}-{frame_end}\n")
    print(f"📝 Created info.txt at {info_path}")

    print("✅ 100% Correct Copying, Renaming, and Cleanup Finished!")


# # === Example usage ===
# source_folder = "./monocular_datasets/monocular_deformable_5/"
# destination_folder = "./monocular_datasets/monocular_deformable_5105/"

# if os.path.exists(destination_folder):
#     shutil.rmtree(destination_folder)
# frame_start = 0
# frame_end = 5
# process_dataset_copy_shift_minus_one_and_rename(source_folder, destination_folder, frame_start, frame_end)


# === Continued Usage for Simulator ===
base_source = "./monocular_datasets/monocular_deformable_5"

stride = 31
length = 33
start_frame = 0
max_frames = 500

# You already have 47 sequences (001–047), so start from 48
count = 1
for start in range(start_frame, max_frames - length + 1, stride):
    end = start + length - 1
    dest_folder = f"{base_source}{count:03d}/"
    print(f"Generating {dest_folder} from {start} to {end}")
    process_dataset_copy_shift_minus_one_and_rename(base_source, dest_folder, start, end)
    count += 1
