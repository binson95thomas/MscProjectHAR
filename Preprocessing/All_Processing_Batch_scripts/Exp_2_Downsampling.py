import os
import subprocess
import sys

def execute_python_scripts(folder_path):
    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    # Filter files to include only Python scripts (files with .py extension)
    python_scripts = [file for file in files if file.endswith('.py') and file not in excluded_files and target in file]
    print(f"Files identified are {python_scripts}")

    user_input = input("Do you want to proceed? (y/n): ")
    if user_input != 'y':
        print("Exiting the Process")
        sys.exit()

    # Execute each Python script
    for script in python_scripts:

        script_path = os.path.join(folder_path, script)
        print(f"Executing script: {script}")
        with open(log_path, 'a') as log_file:
                log_file.write(f"Executing script: {script} \n")
        
        # Use subprocess to run the script
        try:
            subprocess.run(['python', script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")

if __name__ == "__main__":
    folder_path = "./All_DownSample_Scripts/Exp_2"  # Replace with the path to your folder containing Python scripts
    excluded_files = ["nothing.py"]  # Replace with the names of scripts to exclude
    target="Exp_2"

    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    name_only=os.path.splitext(script_name)
    model_folder=name_only[0]
    print(f"ModelName is {model_folder}")

    log_path=f'./All_Batch_scripts/{model_folder}.log'

    execute_python_scripts(folder_path)
    print("All Scripts has been processed.")
    with open(log_path, 'a') as log_file:
        log_file.write("All Scripts has been processed..\n")

    # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
