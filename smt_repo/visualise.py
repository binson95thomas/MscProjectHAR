import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Folder where the numpy arrays are saved
folder_path = 'C:\\Binson\\Cloud\\OneDrive - University of Hertfordshire\\ProjectWorkables\\GPU_DS\\balanced_Canny\\Subject1\\Activity1\\Trial1\\Subject1Activity1Trial1Camera1_2018-07-04T12_04_20.238369.npy'

def load_arrays(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                yield os.path.join(root, file)

def visualize_optical_flow(array_path):
    optical_flows = np.load(array_path)

    fig, ax = plt.subplots()
    im = ax.imshow(optical_flows[0, :, :], cmap='viridis', animated=True)
    ax.axis('off')
    plt.title('Optical Flow Visualization')

    def update_fig(i):
        im.set_array(optical_flows[i, :, :])
        return im,

    ani = animation.FuncAnimation(fig, update_fig, frames=optical_flows.shape[0], blit=True)
    plt.show()

if __name__ == "__main__":
    for array_path in load_arrays(folder_path):
        visualize_optical_flow(array_path)
