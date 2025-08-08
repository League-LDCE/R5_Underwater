# object_detection.py

import cv2
import numpy as np

# --- 1. Load the YOLO model and configuration files ---
# IMPORTANT: Ensure the yolov3.weights, yolov3.cfg, and coco.names
# files are in the same directory as this script.
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: YOLO model files not found. Please download 'yolov3.weights', 'yolov3.cfg', and 'coco.names'.")
    print("Place them in the same directory as this script.")
    exit()

# Get the names of the output layers of the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# --- 2. Initialize the webcam ---
# The argument 0 specifies the default laptop camera.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam. Check if a camera is connected or if another program is using it.")
    exit()

print("Webcam started. Press 'q' to quit.")

# --- 3. Main loop for capturing and processing video frames ---
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    height, width, channels = frame.shape

    # Pre-process the frame for the YOLO model
    # The blob is a 4D tensor with shape (1, channels, height, width)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform a forward pass through the network to get the detections
    outs = net.forward(output_layers)

    # --- 4. Process the detections and draw bounding boxes ---
    class_ids = []
    confidences = []
    boxes = []

    # Iterate over the output layers to find detections
    for out in outs:
        for detection in out:
            # The first 5 values are the center_x, center_y, width, height, and objectness score
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions (low confidence)
            if confidence > 0.5:
                # Scale the bounding box coordinates to the original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the top-left coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Ensure indexes is not empty before drawing
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # Draw the bounding box and label
            color = (0, 255, 0) # Green color for the box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- 5. Display the result ---
    cv2.imshow("Object Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()