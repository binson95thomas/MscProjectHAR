import os
import shutil

def count_files_and_size(root_folder, search_string):
    file_count = 0
    total_size = 0
    
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            if search_string in filename:
                file_count += 1
                total_size += os.path.getsize(file_path)
                
    return file_count, total_size

def copy_files(root_folder, search_string, destination_folder):
    file_count, total_size = count_files_and_size(root_folder, search_string)
    
    print(f"Total files found: {file_count}")
    print(f"Total size of files: {total_size/(1024*1024*1024)} GB")
    
    copy_option = input("Do you want to copy all these files? (yes/no): ")
    if copy_option.lower() == "yes":
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                if search_string in filename:
                    shutil.copy(file_path, destination_folder)
                    print(f"File '{filename}' copied to '{destination_folder}'")
    else:
        print("No files copied.")

# Example usage
root_folder = "H:\\Binson\\NTU_RGB_Extracted"
search_string = "A042"
destination_folder = "H:\\Binson\\NTU_Falls_only"

copy_files(root_folder, search_string, destination_folder)
