import os
import json
from rich import print
from copy import deepcopy
from tqdm import tqdm
import pandas as pd


def find_json_files(directory, json_files=None):
    """
    Recursively searches a directory for JSON files.

    Parameters:
    - directory (str): The directory to search.
    - json_files (list): Internal use for collecting JSON file paths (default is None).

    Returns:
    - list: A list of paths to all JSON files found in the directory and its subdirectories.
    """
    if json_files is None:
        json_files = []
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return json_files

    # List all items in the current directory
    try:
        entries = os.listdir(directory)
    except PermissionError:
        print(f"Permission denied to access '{directory}'.")
        return json_files

    for entry in entries:
        entry_path = os.path.join(directory, entry)

        # If entry is a file and ends with '.json', add to the list
        if os.path.isfile(entry_path) and entry_path.endswith('.json'):
            json_files.append(entry_path)
        
        # If entry is a directory, recurse into it
        elif os.path.isdir(entry_path):
            find_json_files(entry_path, json_files)
    
    return json_files


def read_json_to_dict(json_file_path):
    """
    Reads a JSON file and stores its contents in a dictionary.

    Parameters:
    - json_file_path (str): The path to the JSON file.

    Returns:
    - dict: A dictionary containing the JSON data.
    - None: If an error occurs (e.g., file not found, invalid JSON).
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)  # Parse JSON file to a dictionary
        return data
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file '{json_file_path}'.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


def get_action_mappings(json_dict):
    actions_dict = {}
    for index in json_dict['openlabel']['actions']:
        action = json_dict['openlabel']['actions'][index]['type']
        actions_dict[index] = action
    return actions_dict


def get_frames_and_labels(json_dict, actions_map):
    data = json_dict['openlabel']['frames']
    frames_dict = {}
    for frame in data:
        try:
            actions = data[frame]['actions']
            actions_list = []
            for action in actions:
                act = actions_map[action]
                actions_list.append(act)
            frame = int(frame)
            frames_dict[frame] = {}
            frames_dict[frame]['actions'] = actions_list
        except Exception as e:
            print(f"\nUnable to get label for frame {frame}: {data[frame]}")
    return frames_dict


def offset_frames(json_dict, frame_labels):
    data = json_dict['openlabel']['streams']
    face_offset = data['face_camera']['stream_properties']['sync']['frame_shift']
    body_offset = data['body_camera']['stream_properties']['sync']['frame_shift']
    hands_offset = data['hands_camera']['stream_properties']['sync']['frame_shift']
    frame_info = deepcopy(frame_labels)
    
    for index in frame_labels:
        face_index = index - face_offset
        body_index = index - body_offset
        hands_index = index - hands_offset 
        if face_index >= 0:
            frame_info[index]['face_frame'] = f"face_frame_{face_index:06d}.jpg"
        if body_index >= 0:
            frame_info[index]['body_frame'] = f"body_frame_{body_index:06d}.jpg"
        if hands_index >= 0:
            frame_info[index]['hands_frame'] = f"hands_frame_{hands_index:06d}.jpg"

    return frame_info


def process_json_files(json_files):
    """
    Process a list of JSON files to extract and transform frame data into a list-like format.
    
    Parameters:
        json_files (list): List of file paths to JSON files.
        read_json_to_dict (func): Function to read a JSON file and convert it to a dictionary.
        get_action_mappings (func): Function to map actions from the JSON data.
        get_frames_and_labels (func): Function to extract frames and their labels.
        offset_frames (func): Function to offset frame data.
    
    Returns:
        list: Transformed data from all JSON files in a list-like format.
    """
    files_list = []
    
    for file in tqdm(json_files, desc="Processing JSON files"):
        # Read JSON file into dictionary
        json_data = read_json_to_dict(file)
        
        # Extract mappings and frame data
        actions_map = get_action_mappings(json_data)
        frame_labels = get_frames_and_labels(json_data, actions_map)
        frame_info = offset_frames(json_data, frame_labels)

        new_fname = file.split("\\")[-1] #"/".join(file.split("\\")[1:])
        new_fname = "_".join(new_fname.split("_")[:-3])
        
        # Transform dictionary into rows
        rows = []
        for frame_id, frame_data in frame_info.items():
            row = {
                'file': new_fname,
                'frame_id': int(frame_id),
                'actions': ", ".join(frame_data['actions']) if frame_data['actions'] else None,
                'face_frame': frame_data.get('face_frame', None),
                'body_frame': frame_data.get('body_frame', None),
                'hands_frame': frame_data.get('hands_frame', None),
            }
            rows.append(row)
        
        files_list.extend(rows)
    
    return files_list

def segment_rows_by_file_action(df, file_col="file", action_col="actions", frame_col="frame_id"):
    """
    Labels rows with segment IDs based on file, action, and consecutive frames.

    Parameters:
    - df: DataFrame containing the data.
    - file_col: Column name for the file.
    - action_col: Column name for the action.
    - frame_col: Column name for the frame sequence.

    Returns:
    - DataFrame with an added 'segment_id' column.
    """
    # Sort the DataFrame by file, action, and frame_id to ensure order
    df = df.sort_values(by=[file_col, action_col, frame_col]).reset_index(drop=True)
    
    # Initialize the segment ID
    segment_id = 0
    segment_ids = [segment_id]  # List to store segment IDs for each row
    
    # Iterate over rows to assign segment IDs
    for i in range(1, len(df)):
        # Check if the current row belongs to the same segment as the previous row
        same_file = df.loc[i, file_col] == df.loc[i - 1, file_col]
        same_action = df.loc[i, action_col] == df.loc[i - 1, action_col]
        consecutive_frame = df.loc[i, frame_col] == df.loc[i - 1, frame_col] + 1
        
        if same_file and same_action and consecutive_frame:
            # Same segment as previous row
            segment_ids.append(segment_id)
        else:
            # Start a new segment
            segment_id += 1
            segment_ids.append(segment_id)
    
    # Add the segment IDs to the DataFrame
    df["segment_id"] = segment_ids
    
    return df

def one_hot_encode_actions(df, column_name="actions"):
    """
    One-hot encodes the actions in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the actions.
    - column_name (str): The name of the column containing the actions (default is "actions").

    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded actions.
    """
    # Step 1: Split the column values into lists
    df["actions_split"] = df[column_name].apply(lambda x: x.split(", "))
    
    # Step 2: Flatten all unique actions
    unique_actions = set(action for actions_list in df["actions_split"] for action in actions_list)
    
    # Step 3: Create one-hot encoded columns for each unique action
    for action in unique_actions:
        df[action] = df["actions_split"].apply(lambda x: 1 if action in x else 0)
    
    # Step 4: Drop intermediate column (optional)
    df = df.drop(columns=["actions_split"])

    return df

if __name__ == "__main__":
    ROOT_DIR = "../DMD"
    OUTPUT_PARQUET = "labels.parquet"

    json_files = find_json_files(ROOT_DIR)
    data = process_json_files(json_files)
    df = pd.DataFrame(data)
    # Apply the function to the labels DataFrame
    df["yawning"] = df["actions"].apply(lambda x: "yawning" in str(x).lower())
    df = segment_rows_by_file_action(df, file_col="file", action_col="yawning", frame_col="frame_id")
    df = df.sort_values(by=["file", "frame_id"]).reset_index(drop=True)
    df = one_hot_encode_actions(df)
    df.to_parquet(OUTPUT_PARQUET, index=False)