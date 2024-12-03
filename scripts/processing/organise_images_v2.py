import os
import shutil
import logging
from tqdm import tqdm
import pandas as pd

SOURCE_DIR = "./processed"

def copy_file(src, dst, file):
    """Copy a file with metadata preservation, creating destination directory if needed."""
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
    shutil.copy2(f"{src}/{file}", f"{dst}/{file}")

def load_labels(parquet_file):
    """Load labels from a Parquet file."""
    return pd.read_parquet(parquet_file)

# def create_dirs(base_dir):
#     """Create necessary directories in bulk to avoid repetitive calls."""
#     types = ["rgb_body", "rgb_face", "rgb_hands", "rgb_mosaic", "depth_body", "depth_face", 
#              "depth_hands", "ir_body", "ir_face", "ir_hands"]
#     splits = ["train", "val", "test"]
#     labels = ["pos", "neg"]
    
#     dirs_to_create = [
#         os.path.join(base_dir, type_value, split, label)
#         for type_value in types
#         for split in splits
#         for label in labels
#     ]
    
#     for dir_path in dirs_to_create:
#         os.makedirs(dir_path, exist_ok=True)

def organise(df):
    """Organize image files based on DataFrame information."""
    # create_dirs(SOURCE_DIR)
    
    # Predefine split mapping and cache paths
    split_dict = {i: "train" for i in range(1, 11)}
    split_dict.update({11: "val", 12: "val", 13: "test", 14: "test", 15: "test"})
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organising images"):
        if pd.notna(row["face_frame"]):
            type_value = "rgb_face" # Do only for rgb_face
            split_value = int(row["file"].split("_")[1])
            split = split_dict.get(split_value, "unknown")
            segment_id = int(row["segment_id"])
            
            # Construct filename and destination
            filename = f"{row['file']}_{type_value}_{'_'.join(row['face_frame'].split('_')[1:])}"
            label = "pos" if row["yawning/Yawning without hand"] or row["yawning/Yawning with hand"] else "neg"
            
            src_path = os.path.join(SOURCE_DIR, split)
            dst_path = os.path.join(SOURCE_DIR, type_value, split, label, str(segment_id))
            
            # Copy file with metadata
            copy_file(src_path, dst_path, filename)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading labels")
    df = load_labels("labels.parquet")
    logging.info("Organising images")
    organise(df)
    logging.info("Done")
