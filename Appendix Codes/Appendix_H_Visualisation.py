import traceback

import cv2
import os
import numpy as np
import csv
import re
from timeit import default_timer as timer
from datetime import datetime

#Visualising the preprocessde frame into Grids
def visualise_experiments(self,image_paths,variable_names,grid_height=2,grid_width=3):
    desired_size = (512, 384)
    # grid_height, grid_width = 2, 4
    grid_image = np.zeros((grid_height * desired_size[1], grid_width * desired_size[0], 3), dtype=np.uint8)

    for i,(variable_name, image) in enumerate(zip(variable_names, image_paths)):
        
        # Resize the image to a fixed size
        resized_image = cv2.resize(image, desired_size)
        if len(resized_image.shape) == 2:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        # Calculate the row and column index for the grid
        row = i // grid_width
        col = i % grid_width
        # Place the resized image into the grid
        grid_image[row * desired_size[1]:(row + 1) * desired_size[1], col * desired_size[0]:(col + 1) * desired_size[0], :] = resized_image 
        label = variable_name
        cv2.putText(grid_image, label, (col * desired_size[0] + 10, (row + 1) * desired_size[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Display the grid using cv2.imshow()
    cv2.imshow("Image Grid", grid_image)
    cv2.waitKey(1)  

#Following code generates all the experiment output using code of appendix E
# and pass the experiment frames to visualise_experiments
def compute_edge_detector(self, current_frame):
    current_frame = cv2.normalize(
        src=current_frame,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    fgmask_curr = self.fgbg.apply(current_frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
    current_frame_masked = cv2.bitwise_and(
        current_frame, current_frame, mask=fgmask_curr
    )
    exp_1 = cv2.Canny(current_frame_masked, 65, 195)


    edge_frame_curr =cv2.Canny(current_frame,65, 195)
    fgmask_curr = self.fgbg.apply(current_frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
    exp_2 = cv2.bitwise_and(
        edge_frame_curr, edge_frame_curr, mask=fgmask_curr
    )

    fgmask_curr = self.fgbg.apply(current_frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
    current_frame_masked = cv2.bitwise_and(
        current_frame, current_frame, mask=fgmask_curr
    )
    exp_3 = np.uint8(cv2.Laplacian(current_frame_masked,cv2.CV_64F))


    edge_frame_curr =np.uint8(cv2.Laplacian(current_frame,cv2.CV_64F))
    fgmask_curr = self.fgbg.apply(current_frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
    exp_4 = cv2.bitwise_and(
        edge_frame_curr, edge_frame_curr, mask=fgmask_curr
    )

    fgmask_curr = self.fgbg.apply(current_frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
    current_frame_masked = cv2.bitwise_and(
        current_frame, current_frame, mask=fgmask_curr
    )
    exp_5 =cv2.Sobel(src=current_frame_masked, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) 


    edge_frame_curr =cv2.Sobel(src=current_frame, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) 
    fgmask_curr = self.fgbg.apply(current_frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
    fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
    exp_6 = cv2.bitwise_and(
        edge_frame_curr, edge_frame_curr, mask=fgmask_curr
    )
    # Load your images (replace these file paths with your own images)
    exp_5 = exp_5.astype(np.float32)  # Convert to float32
    exp_5 = cv2.convertScaleAbs(exp_5)  # Convert to CV_8U  
    exp_6 = exp_6.astype(np.float32)  # Convert to float32
    exp_6 = cv2.convertScaleAbs(exp_6)  # Convert to CV_8U  
    image_paths = [current_frame,exp_1,exp_2,exp_3,exp_4,exp_5,exp_6]
    # image_paths = [cexp_1,exp_2,exp_3,exp_4,exp_5,exp_1]
    image_paths2=[exp_5,exp_6]
    variable_names = ["Original","Experiment 1.1", "Experiment 1.2", "Experiment 1.3", "Experiment 1.4", "Experiment 1.5", "Experiment 1.6"]
    variable_names2 = ["Experiment 5.1", "Experiment 6.1"]

    self.visualise_experiments(image_paths,variable_names,grid_height=2,grid_width=4)

    resized_frame = cv2.resize(exp_1, (51, 38))
    return resized_frame

