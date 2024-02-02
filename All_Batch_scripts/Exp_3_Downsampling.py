import os
import subprocess

def execute_python_scripts(folder_path):
    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    # Filter files to include only Python scripts (files with .py extension)
    python_scripts = [file for file in files if file.endswith('.py')]

    # Execute each Python script
    for script in python_scripts:
        script_path = os.path.join(folder_path, script)
        print(f"Executing script: {script}")
        
        # Use subprocess to run the script
        try:
            subprocess.run(['python', script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")

if __name__ == "__main__":
    folder_path = "./All_DownSample_Scripts/Exp_3"  # Replace with the path to your folder containing Python scripts
    execute_python_scripts(folder_path)
