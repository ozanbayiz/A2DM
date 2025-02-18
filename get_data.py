import os
import sys
import glob
import gdown
import zipfile
import concurrent.futures
import numpy as np

def download_slahmr_data(output_path=None):
    """ download SLAHMR data from google drive """
    file_url = f'https://drive.google.com/uc?export=download&id=16XIl-C9pEbsEF6vE8RW_6F3os4doIgER'
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    gdown.download(file_url, output_path, quiet=False)
    with zipfile.ZipFile("slahmr.zip", 'r') as zip_ref:
        zip_ref.extractall(output_path)  
    os.remove("slahmr.zip")

def is_valid_npz(file_path):
    """
    Returns a tuple (is_valid, message) where is_valid is True if the .npz file has keys
    'trans', 'root_orient', and 'pose_body', and if the time dimension (axis 1) of each array is 100.
    Otherwise, returns False along with a message describing the problem.
    """
    required_keys = ['trans', 'root_orient', 'pose_body']
    try:
        data = np.load(file_path)
    except Exception as e:
        return False, f"Error loading file: {e}"

    for key in required_keys:
        if key not in data:
            return False, f"Missing key '{key}'"
        arr = data[key]
        if len(arr.shape) < 2 or arr.shape[1] != 100 or arr.shape[0] != 2:
            return False, f"Key '{key}' has shape {arr.shape} (expected time dimension = 100 and 2 persons)"
    return True, None

def check_file(file_path):
    valid, message = is_valid_npz(file_path)
    return file_path, valid, message

def remove_invalid_npz_parallel(directory, max_workers=8):
    npz_files = glob.glob(os.path.join(directory, "*.npz"))
    print(f"Found {len(npz_files)} .npz files in '{directory}'.")
    files_to_remove = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_file, file_path): file_path for file_path in npz_files}
        for future in concurrent.futures.as_completed(futures):
            file_path, valid, message = future.result()
            if not valid:
                print(f"File {file_path} is invalid: {message}")
                files_to_remove.append(file_path)

    removed_count = 0
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
            removed_count += 1
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")
    print(f"Removed {removed_count} invalid file(s).")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        output_path = None
    else:
        output_path = sys.argv[1]
        
    download_slahmr_data(output_path)
    os.rename("slahmr", "sns_slahmr")
    data_dir = os.path.join(output_path, "sns_slahmr") if output_path else "sns_slahmr"
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a valid directory.")
        sys.exit(1)
    remove_invalid_npz_parallel(data_dir)
