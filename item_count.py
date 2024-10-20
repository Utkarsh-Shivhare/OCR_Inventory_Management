

from ultralytics import YOLO

# Load the YOLO model (make sure to replace the path with the correct one for your model)
model = YOLO('Item_count_yolov8.pt')  # e.g., 'best.pt'

# Specify the image for testing (replace 'image.jpg' with your image file)
image_path = 'test_image.jpg' #place the image path here

# Run predictions on the image
results = model.predict(source=image_path, imgsz=1080, device='cpu', conf=0.25, save=True)

# Count the number of items detected
num_detected_items = len(results[0].boxes)  # 'boxes' holds the detected objects

# Print the count of detected items
print(f"Number of items detected: {num_detected_items}")

# Optionally, display the image with bounding boxes
for result in results:
    result.show()  # Shows the output with bounding boxes and labels
