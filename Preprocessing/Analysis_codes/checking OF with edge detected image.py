import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or change to another index if you have multiple cameras

# Initialize the parameters for the Farneback optical flow algorithm
farneback_params = dict(
    winSize=(15, 15),
    levels=3,
    iterations=5,
    polyN=7,
    polySigma=1.2,
    flags=0
)

# Read the first frame
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 300)
    edges_prev = cv2.Canny(prev_frame_gray, 100, 300)

    # Originaal
    # Compute optical flow using the Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, gray, None,0.5, 1, 15, 3, 5, 1.2, 0)

    # Compute the polar coordinates and magnitude of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an RGB image representation of the optical flow
    hsv = np.zeros_like(frame)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
    hsv[..., 1] = 255  # Saturation is set to maximum
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude

    optical_flow_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # concatenated_frames = cv2.hconcat([gray, optical_flow_result])

    cv2.imshow('Optical Flow Original', optical_flow_result)

#--------------------------------------------------------------------------------------------------
    flow2 = cv2.calcOpticalFlowFarneback(edges_prev, edges, None,0.5, 1, 15, 3, 5, 1.2, 0)

    # Compute the polar coordinates and magnitude of the flow vectors
    magnitude2, angle2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

    # Create an RGB image representation of the optical flow
    hsv2 = np.zeros_like(frame)
    hsv2[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
    hsv2[..., 1] = 255  # Saturation is set to maximum
    hsv2[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude

    optical_flow_result2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    # concatenated_frames2 = cv2.hconcat([gray, optical_flow_result])

    cv2.imshow('Optical Flow Edges', optical_flow_result2)

    concatenated_frames = cv2.hconcat([gray, edges])

    cv2.imshow('Source', concatenated_frames)
    

    # Update the previous frame for the next iteration
    prev_frame_gray = gray.copy()

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
