import cv2
import numpy as np



#Function to compute edge detector for experiment
#Function takes each frame as an input

def compute_edge_detector(self, current_frame):
        current_frame = cv2.normalize(
            src=current_frame,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        #Applying the background subtraction on frame
        fgmask_curr = self.fgbg.apply(current_frame)
        #Kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)
        #Generating the mask
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        #Applying the mask on current frame
        current_frame_masked = cv2.bitwise_and(
            current_frame, current_frame, mask=fgmask_curr
        )
        #performing edge detection
        current_frame_masked = cv2.Canny(current_frame_masked, 65, 195)
        #Resizing the frame to desired resolution
        resized_frame = cv2.resize(current_frame_masked, (51, 38))
        return resized_frame