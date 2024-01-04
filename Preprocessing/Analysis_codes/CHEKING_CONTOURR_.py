import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for human detection
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans in the frame
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Draw rectangles on the mask based on human detection
    for (x, y, w, h) in humans:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)

    # Apply morphological operations for mask refinement
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result
    cv2.imshow('Masked Video', result)

    if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
