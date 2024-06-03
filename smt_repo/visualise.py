import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import traceback

# Folder where the numpy arrays are saved
folder_path = 'E:\\Outputs\\Exp_6_4_OF_HighRes_128x96\\Unbalanced\\Subject1\\Activity1\\Trial1'

def load_arrays(folder_path):
    print(folder_path)
    for root, dirs, files in os.walk(folder_path):
        print(f'files are{files}')
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
    print('start')
    try:
        print('Vis start')
        for array_path in load_arrays(folder_path):
            visualize_optical_flow(array_path)
        print('Vis complete')
    except Exception as e:
        print(f'Error {e}')
        error_stack = traceback.format_exc()
        print("=======================================================================")
        print(f"Error stack:\n{error_stack}")
