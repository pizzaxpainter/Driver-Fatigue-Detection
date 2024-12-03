import os
import shutil
import logging
from PIL import Image, ImageFilter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ================== Configuration ==================

# Paths
OUTPUT_DIR = "./outputs"            # Directory with original image sequences
PROCESSED_DIR = "./processed"       # Directory to save resized images and organize splits

# Image settings
TARGET_SIZE = (256, 256)            # Target size for images

# ================== Logging Configuration ==================

logging.basicConfig(
    filename='processing_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# ================== Step 1: Resize Images Recursively (Optimized) ==================

def resize_and_pad_image(input_path, output_path, target_size):
    """
    Resize an image to fit within target_size while maintaining aspect ratio.
    Pads the image with black borders if necessary.
    Applies a sharpening filter after resizing to enhance details.
    Saves the image with high quality settings to minimize compression artifacts.
    """
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            # Determine the appropriate resampling filter
            try:
                # For Pillow >= 10.0.0
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                try:
                    # For Pillow < 10.0.0
                    resample = Image.LANCZOS
                except AttributeError:
                    # Fallback to BICUBIC if LANCZOS is not available
                    resample = Image.BICUBIC
            
            # Resize the image while maintaining aspect ratio
            img.thumbnail(target_size, resample)
            
            # Create a new image with a black background
            new_img = Image.new("RGB", target_size, (0, 0, 0))
            
            # Calculate padding to center the image
            paste_x = (target_size[0] - img.width) // 2
            paste_y = (target_size[1] - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            # Apply a sharpening filter to enhance details
            sharpened_img = new_img.filter(ImageFilter.SHARPEN)
            
            # Save the image with high quality to minimize compression loss
            sharpened_img.save(output_path, "JPEG", quality=100, optimize=True)
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")


def resize_images(output_dir, processed_dir, target_size):
    """
    Recursively traverse the output directory, resize images in parallel, and save them to the processed directory,
    maintaining the same directory structure.
    
    Args:
        output_dir (str): Path to the directory with original image sequences.
        processed_dir (str): Path to the directory where resized images will be saved.
        target_size (tuple): Target size for the resized images (width, height).
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Collect all image file paths with their relative paths
    image_tasks = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_image_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_image_path, output_dir)
                dest_image_path = os.path.join(processed_dir, relative_path)
                dest_image_dir = os.path.dirname(dest_image_path)
                if not os.path.exists(dest_image_dir):
                    os.makedirs(dest_image_dir, exist_ok=True)
                image_tasks.append((src_image_path, dest_image_path))
    
    print("Resizing images using parallel processing...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Prepare a partial function with fixed target_size
        resize_func = partial(resize_and_pad_image, target_size=target_size)
        # Submit all tasks to the executor
        futures = {executor.submit(resize_func, src, dest): (src, dest) for src, dest in image_tasks}
        # Use tqdm to display progress
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            pass  # Progress is handled by tqdm

# ================== Step 2: Split Dataset Based on Directory Number ==================

def split_dataset(processed_dir):
    """
    Split the dataset into train, val, and test sets based on directory numbers.
    Assign directories to splits as follows:
        - Number <= 10: train
        - Number 11-12: val
        - Number 13-15: test

    The split directories are created within the processed_dir and contain all JPEG/PNG images
    directly without preserving the original subdirectory structure.

    Args:
        processed_dir (str): Path to the directory with resized images.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Define split assignments
    split_assignment = {}
    for num in range(1, 16):
        if num <= 10:
            split_assignment[num] = 'train'
        elif num in [11, 12]:
            split_assignment[num] = 'val'
        elif num in [13, 14, 15]:
            split_assignment[num] = 'test'
        else:
            split_assignment[num] = 'train'  # Default to train for any other numbers

    # Define the paths for split directories
    split_dirs = {
        'train': os.path.join(processed_dir, 'train'),
        'val': os.path.join(processed_dir, 'val'),
        'test': os.path.join(processed_dir, 'test')
    }

    # Create split directories if they don't exist
    for split, path in split_dirs.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logging.info(f"Created directory: {path}")

    # Traverse the first two levels: group and number
    groups = [
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d)) and d not in {"train", "val", "test"}
    ]

    logging.info("Splitting dataset into train, val, and test based on directory numbers...")
    for group in tqdm(groups, desc="Processing Groups", unit="group"):
        group_path = os.path.join(processed_dir, group)
        numbers = [
            d for d in os.listdir(group_path)
            if os.path.isdir(os.path.join(group_path, d))
        ]

        for number in tqdm(numbers, desc=f"Processing {group}", unit="directory", leave=False):
            try:
                num = int(number)
                split = split_assignment.get(num, 'train')  # Default to 'train' if not found
            except ValueError:
                logging.error(f"Directory name '{number}' is not a valid number. Skipping.")
                continue

            src_number_dir = os.path.join(group_path, number)
            dest_split_dir = split_dirs[split]

            # Traverse all subdirectories within the number directory and copy files
            for root, dirs, files in os.walk(src_number_dir):
                for file in tqdm(files, desc=f"Copying Files in {number}", unit="file", leave=False):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(dest_split_dir, file)

                        # Handle potential filename conflicts by appending a unique identifier
                        if os.path.exists(dest_file):
                            base, ext = os.path.splitext(file)
                            counter = 1
                            while True:
                                new_filename = f"{base}_{counter}{ext}"
                                dest_file = os.path.join(dest_split_dir, new_filename)
                                if not os.path.exists(dest_file):
                                    break
                                counter += 1

                        try:
                            shutil.copy2(src_file, dest_file)
                        except Exception as e:
                            logging.error(f"Error copying {src_file} to {dest_file}: {e}")

    logging.info("Dataset splitting completed successfully.")

# ================== Main Execution ==================

def main():
    """
    Execute all steps: Resize images, split dataset, and convert to LMDB.
    Skips steps if their outputs already exist.
    """
    # Step 1: Resize Images (Parallelized)
    if not os.path.exists(PROCESSED_DIR) or not any(os.scandir(PROCESSED_DIR)):
        print("Step 1: Resizing images...")
        resize_images(OUTPUT_DIR, PROCESSED_DIR, TARGET_SIZE)
    else:
        print("Step 1: Resized images already exist. Skipping...")

    # Step 2: Split Dataset Based on Directory Number
    split_subdirs = ['train', 'val', 'test']
    if not all(os.path.exists(os.path.join(PROCESSED_DIR, subdir)) for subdir in split_subdirs):
        print("Step 2: Splitting dataset...")
        split_dataset(PROCESSED_DIR)
    else:
        print("Step 2: Dataset already split. Skipping...")

if __name__ == "__main__":
    main()
