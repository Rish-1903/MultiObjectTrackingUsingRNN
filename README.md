# Vehicle Tracking Using Background Subtraction and RNN

This project demonstrates object tracking in a video using background subtraction and Recurrent Neural Networks (RNN) for predicting object movement. The key components of this project include background subtraction for initial object detection, object tracking with an LSTM-based RNN, and predicting object movements for continuous tracking.

## Features

1. **Background Subtraction**:
   - Uses OpenCV's `BackgroundSubtractorMOG2` for detecting moving objects in the video.
   - Applies a threshold to remove shadows and filters out small objects based on contour area.
   - Objects are tracked across frames and their positions (bounding boxes) are stored for further prediction.

2. **RNN Model for Object Tracking**:
   - A simple LSTM-based RNN model is used to predict the future positions and sizes of detected objects.
   - The model is trained using a sequence of object positions and sizes (bounding box coordinates) over a history of past frames.

3. **Object Tracking Data Preparation**:
   - Object positions (x, y) and sizes (w, h) are extracted from each frame.
   - These data points are grouped into sequences to serve as input for training the RNN.

4. **Tracking Predictions**:
   - Once trained, the RNN model predicts the next positions and sizes of the objects in subsequent frames.
   - The predicted bounding boxes are drawn on the frames for visualization.

5. **Output**:
   - The final tracking is visualized with bounding boxes drawn on each frame.
   - The output video with tracked objects can be saved and reviewed.

## Workflow

1. **Background Subtraction**:
   - The video is processed using background subtraction to identify moving objects.
   - Object contours are detected and their bounding boxes are extracted.

2. **Prepare Data for RNN**:
   - Sequences of object positions and sizes are prepared for RNN training.

3. **Train the RNN**:
   - An LSTM model is trained using past object data to predict future positions and sizes.

4. **Track Objects**:
   - The trained RNN model is used to predict and track objects in the video.

## Requirements

- OpenCV
- TensorFlow
- NumPy

## How to Run

1. Place the video file (`vehicles.mp4`) in the project folder.
2. Run the script, and the background subtraction and tracking will be processed.
3. The output video (`Occlusion1.mp4`) with object tracking will be saved.
4. The final tracked objects will be displayed on the video frames.

## Potential Improvements

- Handle occlusions and interactions between objects more effectively.
- Use advanced object detection models for more accurate bounding box predictions.
- Fine-tune the RNN model for better performance in real-time tracking scenarios.

This approach can be extended to other applications like pedestrian tracking, sports tracking, or surveillance systems.
