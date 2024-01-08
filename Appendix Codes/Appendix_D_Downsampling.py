import os
import numpy as np
import shutil

def downsample_dataset(original_folder, output_folder):
    dataset_info = []
    for root, _, files in os.walk(original_folder):
        for file in files:
            if file.endswith('.npy'):
                print(f"Processing {file} ")
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True).item()
                label = data['label']
                dataset_info.append((file_path, label))
    #Identifying the fall and No Fall samples based on activity id
    # In context with dataset
                #Activity ID - 1-5 is a fall
                #Acitivity ID - 6-11 is a No fall
    fall_samples = [info for info in dataset_info if 1 <= info[1] <= 5]
    non_fall_samples = [info for info in dataset_info if info[1] > 5]
    #Calculating total falls
    num_falls = len(fall_samples)
    print(num_falls)
    #Randomly choosing no fall scenario
    downsampled_non_falls_indices = np.random.choice(len(non_fall_samples), size = num_falls, replace = False)
    downsampled_non_falls = [non_fall_samples[i] for i in downsampled_non_falls_indices]
    
    for file_path, _ in fall_samples + list(downsampled_non_falls):
        print(f"Creating file {file_path} ")
        relative_path = os.path.relpath(file_path, original_folder)
        new_path = os.path.join(output_folder, relative_path)
        
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        
        shutil.copyfile(file_path, new_path)
        
if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    name_only=os.path.splitext(script_name)
    model_type=name_only[0]
    print(f"Downsampling ModelName is {model_type}")
    original_folder = f'../Outputs/{model_type}/Unbalanced'
    output_folder = f'../Outputs/{model_type}/Balanced/'
    with open(f'../Outputs/{model_type}/DS_log.log', 'a') as file:
            file.write(f' Downsampling started for {original_folder} with ou in {output_folder} \n')
    print(f"file is {original_folder} and {output_folder}")
    downsample_dataset(original_folder, output_folder)
    with open(f'../Outputs/{model_type}/DS_log.log', 'a') as file:
        file.write(f' Downsampling Complete for {original_folder} with ou in {output_folder} \n')

