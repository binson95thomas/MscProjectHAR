import os

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        # Exclude files from the loop
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size
def convert_size(size_bytes):
    # Convert bytes to megabytes (MB) and gigabytes (GB)
    mb_size = size_bytes / (1024 * 1024)
    gb_size = size_bytes / (1024 * 1024 * 1024)
    return mb_size, gb_size

def print_folder_structure(root_folder, max_depth=2, current_depth=0):
    if current_depth > max_depth:
        return

    subdirectories = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    for subdir in subdirectories:
        subdir_path = os.path.join(root_folder, subdir)
        folder_size_bytes = get_folder_size(subdir_path)
        mb_size, gb_size = convert_size(folder_size_bytes)

        print(f"{subdir_path} - Size: {mb_size:.2f} MB ({gb_size:.2f} GB)")

        # Recursively call for the next level
        print_folder_structure(subdir_path, max_depth, current_depth + 1)




# Example usage
root_folder_path = "C:\\Binson\\Codes\\Outputs"
print_folder_structure(root_folder_path, max_depth=1)
# print_folder_structure(root_folder_path)