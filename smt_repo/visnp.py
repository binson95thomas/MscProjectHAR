import numpy as np

# Replace this with the path to your numpy file
file_path = 'E:\\Cloud Sync\\OneDrive - University of Hertfordshire\\ProjectWorkables\\GPU_DS\\balanced_Canny\\Subject1\\Activity1\\Trial1\\Subject1Activity1Trial1Camera1_2018-07-04T12_04_20.238369.npy'

# Load the numpy file
data = np.load(file_path, allow_pickle=True)

# Print the contents
print(data)
