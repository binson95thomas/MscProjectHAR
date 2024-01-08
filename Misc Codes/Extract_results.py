import os

def process_files(base_folder, file_name, output_file):
    with open(output_file, 'w') as output:
        for foldername, subfolders, filenames in os.walk(base_folder):
            print(foldername)
            for filename in filenames:
                if filename == file_name:
                    file_path = os.path.join(foldername, filename)
                    with open(file_path, 'r') as input_file:
                        lines = input_file.readlines()
                        last_lines = lines[-2:]
                        print(last_lines)
                        output.write(f"\nFolder: {foldername}\n")
                        output.writelines(last_lines)

# Example usage
base_directory = "./results"
specific_file_name = "Trainlog.log"
output_file_name = "ExtractedTestResults.txt"

process_files(base_directory, specific_file_name, output_file_name)