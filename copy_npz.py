import os
import shutil
import re

def copy_matching_npz_to_flat_dest(source_root, destination_folder, gs_list):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Walk through all subfolders of source_root
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith(".npz"):
                # Try to extract gs number from filename
                match = re.search(r'gs_(\d+)', file)
                if not match:
                    continue
                gs_number = int(match.group(1))
                if gs_number in gs_list:
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(destination_folder, file)
                    shutil.copy2(src_path, dest_path)
                    print(f"✅ Copied {file} → {destination_folder}/")

def process_trajectory(traj_id):
    start = traj_id * 1000
    end = start + 31  # exclusive
    gs_list = [10, 13, 16, 19]

    destination_folder = f"./input/simulator/trajectory_{traj_id}_accel_patched_input"

    for traj_num in range(start, end):
        source_folder = f"./simulator_data/trajectory_{traj_num}/"
        copy_matching_npz_to_flat_dest(source_folder, destination_folder, gs_list)


# # ==== USAGE ====
# for traj_num in range(11000, 11031): # trj 11 train sequences, change to 12 if needed
#     source_folder = f"./hamlyn_data/trajectory_{traj_num}/"
#     destination_folder = f"./input/hamlyn/trajectory_11_accel_patched_input" # change to 12 if needed
#     # destination_folder = f"./input/hamlyn/trajectory_12_accel_patched_input_eval" 
#     gs_list = [10, 13, 16, 19]

#     copy_matching_npz_to_flat_dest(source_folder, destination_folder, gs_list)
