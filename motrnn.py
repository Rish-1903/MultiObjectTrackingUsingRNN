import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Background subtraction and object detection
def background_subtraction(file: str, save: bool=False, output_video_path: str='output_video.mp4'):
    cap = cv2.VideoCapture(file)
    
    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=350, detectShadows=True)
    
    # Get the video frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))
    
    # Object tracking data (for RNN)
    object_tracks = []
    object_track_history = {}  # Store object history
    
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            break
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Threshold to remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_frame_objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum size filter for object detection
                (x, y, w, h) = cv2.boundingRect(contour)
                current_frame_objects.append((x, y, w, h))
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Store object positions for RNN tracking
        object_tracks.append(current_frame_objects)
        
        # Save the output video
        if save:
            out.write(frame)
        
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)
        
        keyboard = cv2.waitKey(30)
        if keyboard == ord('q') or keyboard == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return object_tracks


# 2. RNN Model for object tracking (using LSTM)
def create_rnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4))  # Output: x, y, w, h
    model.compile(optimizer='adam', loss='mse')
    return model


# 3. Prepare data for RNN and train
def prepare_tracking_data(object_tracks, history_length=5):
    X, Y = [], []
    for i in range(len(object_tracks) - history_length):
        seq_x = []
        seq_y = []
        for j in range(history_length):
            frame_objects = object_tracks[i + j]
            seq_x.append([obj[:2] for obj in frame_objects])  # (x, y) positions
            seq_y.append([obj[2:] for obj in frame_objects])  # (w, h) size
        X.append(np.array(seq_x))
        Y.append(np.array(seq_y))
    return np.array(X), np.array(Y)


# 4. Predict object movement using RNN model
def track_objects_with_rnn(model, object_tracks, history_length=5):
    tracked_objects = []
    
    for i in range(history_length, len(object_tracks)):
        frame_objects = object_tracks[i - history_length]
        frame_objects_positions = [obj[:2] for obj in frame_objects]  # (x, y)
        
        # Predict next positions (x, y, w, h) using the RNN model
        pred_positions = model.predict(np.array([frame_objects_positions]))  # Predicting next object movement
        
        # Store predictions
        predicted_objects = []
        for obj, pred in zip(frame_objects, pred_positions[0]):
            x, y, w, h = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
            predicted_objects.append((x, y, w, h))
        
        tracked_objects.append(predicted_objects)
    
    return tracked_objects


# 5. Main function to execute the background subtraction and tracking
def main():
    parent_directory = os.path.dirname(os.getcwd())
    video_file = os.path.join(parent_directory, 'ML Project', 'vehicles.mp4')
    
    # Perform background subtraction and get the tracked object information
    object_tracks = background_subtraction(video_file, save=True, output_video_path="Occlusion1.mp4")
    
    # Prepare data for training RNN (using past 5 frames as input)
    X, Y = prepare_tracking_data(object_tracks, history_length=5)
    
    # Create and train RNN model
    model = create_rnn_model((X.shape[1], X.shape[2]))  # Shape = (history_length, number of objects * 2 for (x, y))
    model.fit(X, Y, epochs=10, batch_size=32)
    
    # Predict object movement on new frames
    tracked_objects = track_objects_with_rnn(model, object_tracks)
    
    # Visualize tracked objects
    cap = cv2.VideoCapture(video_file)
    for i, (frame_objects, predicted_objects) in enumerate(zip(object_tracks[5:], tracked_objects)):
        ret, frame = cap.read()
        if frame is None:
            break
        
        for obj, pred in zip(frame_objects, predicted_objects):
            x, y, w, h = pred
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Tracked Objects', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
