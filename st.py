
import cv2
import numpy as np
# import pygame
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import math
import requests
from kalmanfilter import KalmanFilter
from orange_detector import OrangeDetector  # Replace with your object detector import
import threading
import time


# Flask server URL
FLASK_SERVER_URL = "http://localhost:3000"

# Heartbeat function to notify server that Streamlit app is active
def send_heartbeat():
    try:
        response = requests.post(f"{FLASK_SERVER_URL}/heartbeat")
        st.write(response.json())
    except Exception as e:
        st.write("Error sending heartbeat:", e)

# Start a heartbeat thread
def start_heartbeat():
    def heartbeat_loop():
        while True:
            send_heartbeat()
            time.sleep(5)  # Send heartbeat every 5 seconds

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

# Load video capture
cap = cv2.VideoCapture('ball.mp4')  # Use your video file here

# Load Kalman filter
kf = KalmanFilter()

# Load Orange Detector
od = OrangeDetector()

# # Initialize pygame for sound
# pygame.mixer.init()
# sound = pygame.mixer.Sound("alert.mp3")  # Replace with the actual sound file path

# Initialize variables
previous_x_pred = None
previous_y_pred = None
is_sound_playing = False
last_photo_time = 0
photo_interval = 2  # Time in seconds between capturing photos

# Function to calculate the angle between two points
def calculate_angle(prev_x, prev_y, curr_x, curr_y):
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    angle = math.degrees(math.atan2(dy, dx))  # Calculate angle in degrees
    return angle

# Function to send trigger to Flask server
def send_trigger(lat, lng):
    try:
        response = requests.post(f"{FLASK_SERVER_URL}/trigger", json={"latitude": lat, "longitude": lng})
        st.write("latitude",lat)
        st.write("longitude",lng)
        st.write(response.json())
    except Exception as e:
        st.write("Error sending trigger:", e)

# Function to capture a photo with the angle text
def capture_photo(frame, cx, cy, angle):
    font = cv2.FONT_HERSHEY_SIMPLEX
    angle_text = f"Angle: {angle:.2f}°"
    cv2.putText(frame, angle_text, (cx - 50, cy - 20), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the image with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"photo_{timestamp}.png"
    cv2.imwrite(filename, frame)
    st.write(f"Captured photo: {filename}")

# Streamlit app structure
def main():
    st.title("Object Detection and Tracking with Kalman Filter")

    # Display video frames and graph in Streamlit
    stframe = st.empty()  # Placeholder for video frames
    graph_placeholder = st.empty()  # Placeholder for matplotlib graph

    start_heartbeat()  # Start sending heartbeat

    global previous_x_pred, previous_y_pred, is_sound_playing, last_photo_time

    x_data_initial = []
    y_data_initial = []
    x_data_predicted = []
    y_data_predicted = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect object (orange in this case)
        orange_bbox = od.detect(frame)
        if orange_bbox is not None:
            x, y, x2, y2 = orange_bbox
            cx = int((x + x2) / 2)
            cy = int((y + y2) / 2)

            # Predict next position using Kalman filter
            predicted = kf.predict(cx, cy)

            # Update positions and capture photo if moving down
            current_time = datetime.now().timestamp()
            if previous_y_pred is not None and predicted[1] > previous_y_pred:
                live_lat = 19.0760  # Replace with actual latitude
                live_lng = 72.8777  # Replace with actual longitude
                send_trigger(live_lat, live_lng)

                if current_time - last_photo_time >= photo_interval:
                    angle = calculate_angle(previous_x_pred, previous_y_pred, predicted[0], predicted[1])
                    capture_photo(frame, cx, cy, angle)
                    last_photo_time = current_time



            # Draw circles and lines on the frame
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)  # Blue dot for initial position
            cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 10, (0, 0, 255), -1)  # Red dot for predicted position

            # Update graph data
            x_data_initial.append(cx)
            y_data_initial.append(cy)
            x_data_predicted.append(predicted[0])
            y_data_predicted.append(predicted[1])

            # Update previous positions
            previous_x_pred = predicted[0]
            previous_y_pred = predicted[1]

        # Display updated frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Update graph with matplotlib
        fig, ax = plt.subplots()
        ax.set_title("Position Graph")
        ax.plot(x_data_initial, y_data_initial, 'bo-', label="Initial Position")
        ax.plot(x_data_predicted, y_data_predicted, 'ro-', label="Predicted Position")
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 480)
        ax.legend()
        graph_placeholder.pyplot(fig)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
