#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install ultralytics


# In[ ]:


from ultralytics import YOLO

# Initialize the YOLO model with pre-trained weights for detection
model = YOLO('yolov8n.pt')  # Assuming you are using YOLOv8 small model weights

# Train the model with your custom dataset
model.train(data=r"C:\Users\kuksh\ProjectR\project1.yaml", epochs=70)


# In[ ]:





# In[ ]:





# In[ ]:


2


# In[3]:



import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# Train the model with your custom dataset (if not already trained)
# model.train(data="/content/archive/data2.yaml", epochs=3)
model = YOLO('yolov8n.pt') 
# Evaluate model performance on the validation set
metrics = model.val(data=r"C:\Users\kuksh\ProjectR\project1.yaml")


# Predict on an image
image_path = r"C:\Users\kuksh\ProjectR\test\images\Iloveyou-1064ea8d-82f1-11ef-ae4a-b0dcefc488c5_jpg.rf.32ea519cc8e728560613c030abfe3dad.jpg"
results = model(image_path)

# Extract bounding box coordinates and their centers
bounding_boxes = results[0].boxes.xyxy.numpy()  # Extract bounding boxes as numpy array

# Load the image using OpenCV
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the image with bounding boxes
plt.figure(figsize=(12, 8))
plt.imshow(image)
for box in bounding_boxes:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Draw the bounding box
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none'))

    # Draw the center point
    plt.plot(center_x, center_y, 'bo')  # 'bo' stands for blue color and circle marker

    # Print bounding box coordinates and center
    print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")
    print(f"Center: ({center_x}, {center_y})")

plt.axis('off')
plt.show()


# In[7]:


# Evaluate model performance on validation set
metrics = model.val()

# Predict on an image
results = model(r"C:\Users\kuksh\ProjectR\test\images\Iloveyou-1064ea8d-82f1-11ef-ae4a-b0dcefc488c5_jpg.rf.32ea519cc8e728560613c030abfe3dad.jpg")


# In[1]:


cap = cv2.VideoCapture(0)  # '0' is typically the default webcam

while True:  # Loop to continuously capture frames
    ret, frame = cap.read()  # Capture frame from the webcam
    if not ret:
        break  # Break the loop if no frame is captured

    results = model(frame)  # Perform detection on the current frame
    bounding_boxes = results[0].boxes.xyxy.numpy()  # Extract bounding boxes as numpy array

    # Draw the bounding boxes and center points on the frame
    for box in bounding_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # Green box

        # Draw the center point
        cv2.circle(frame, (center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)  # Blue center

        # Print bounding box coordinates and center
        print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")
        print(f"Center: ({center_x}, {center_y})")

    # Display the frame with bounding boxes
    cv2.imshow('YOLO Object Detection', frame)  # Show the frame in a window

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop when 'q' is pressed
        break  # Exit the while loop

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




