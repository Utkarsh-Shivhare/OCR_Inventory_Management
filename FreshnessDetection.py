import cv2
import numpy as np
from roboflow import Roboflow

# Initialize Roboflow
# do not use this key please make an account for api key in RoboFlow
rf = Roboflow(api_key="RBMCiagFraeIHPvptwcS")
project = rf.workspace().project("freshness-fruits-and-vegetables")
model = project.version(7).model

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    try:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform inference
        results = model.predict(frame, confidence=40, overlap=30).json()

        # Extract bounding boxes and labels from the predictions
        for prediction in results['predictions']:
            x, y, w, h = (
                prediction['x'], 
                prediction['y'], 
                prediction['width'], 
                prediction['height']
            )
            class_name = prediction['class']
            confidence = prediction['confidence']
            
            # Calculate bounding box coordinates
            x_min = int(x - w / 2)
            y_min = int(y - h / 2)
            x_max = int(x + w / 2)
            y_max = int(y + h / 2)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Create label with class and confidence score
            label = f"{class_name}: {confidence:.2f}"
            
            # Put the label above the bounding box
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with annotations
        cv2.imshow("Freshness Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
