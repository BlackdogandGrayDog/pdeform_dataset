from infer import compute_dense_flow_for_sequence
from simulator_copy import generate_sequences
from simulator_dataset import auto_generate_datasets
import os



if __name__ == "__main__":
    base_root = "./mono_datasets/monocular_deformable_1/"
    base_root = base_root.rstrip("/")
    compute_dense_flow_for_sequence(base_root)
    length = 23

    generate_sequences(
        base_source=base_root,
        stride=length-2,
        length=length,
        start_frame=0,
        max_frames=700,
    )

    # Optionally batch-generate datasets from detected trajectory folders
    auto_generate_datasets(
        base_source=os.path.dirname(base_root) + "/",
        start_frame_index=2,
        end_frame_index=length-1,
        patch_size=6,
        train_eval="train",
        extra_grid_size=20,
    )
