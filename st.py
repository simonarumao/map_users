import cv2
import numpy as np
import streamlit as st
from datetime import datetime
import math
import requests
from kalmanfilter import KalmanFilter
from orange_detector import OrangeDetector  # Replace with your object detector import

# Flask server URL
FLASK_SERVER_URL = "http://127.0.0.1:5000/trigger"

# Streamlit setup
st.title("Object Tracking and Detection with Kalman Filter")
st.sidebar.title("Settings")

# Initialize Kalman filter and Orange Detector
kf = KalmanFilter()
od = OrangeDetector()

# Sidebar options
photo_interval = st.sidebar.slider("Photo Interval (seconds)", min_value=1, max_value=10, value=2)
enable_sound = st.sidebar.checkbox("Enable Sound Alerts", value=True)

# Initialize variables
previous_x_pred, previous_y_pred = None, None
is_sound_playing, last_photo_time = False, 0

# Initialize graph data
x_data_initial, y_data_initial = [], []
x_data_predicted, y_data_predicted = [], []

# Streamlit elements
frame_placeholder = st.empty()
graph_placeholder = st.empty()

# Load alert sound
audio_file = open("alert.mp3", "rb")  # Replace with your sound file path
audio_bytes = audio_file.read()

# Function to calculate angle
def calculate_angle(prev_x, prev_y, curr_x, curr_y):
    dx, dy = curr_x - prev_x, curr_y - prev_y
    return math.degrees(math.atan2(dy, dx))

# Function to send trigger to Flask server
def send_trigger(lat, lng):
    try:
        response = requests.post(FLASK_SERVER_URL, json={"latitude": lat, "longitude": lng})
        st.sidebar.write(response.json())
    except Exception as e:
        st.sidebar.error(f"Error sending trigger: {e}")

# Function to capture a photo
def capture_photo(frame, cx, cy, angle):
    font = cv2.FONT_HERSHEY_SIMPLEX
    angle_text = f"Angle: {angle:.2f}°"
    cv2.putText(frame, angle_text, (cx - 50, cy - 20), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"photo_{timestamp}.png"
    cv2.imwrite(filename, frame)
    st.sidebar.success(f"Captured photo: {filename}")

# Start video capture
cap = cv2.VideoCapture('ball.mp4')

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read from video.")
        break

    # Detect object
    orange_bbox = od.detect(frame)
    if orange_bbox:
        x, y, x2, y2 = orange_bbox
        cx, cy = int((x + x2) / 2), int((y + y2) / 2)

        # Predict next position
        predicted = kf.predict(cx, cy)

        # Check for movement and capture photo
        current_time = datetime.now().timestamp()
        if previous_y_pred is not None and predicted[1] > previous_y_pred:
            live_lat, live_lng = 19.0760, 72.8777  # Replace with actual coordinates
            send_trigger(live_lat, live_lng)

            if current_time - last_photo_time >= photo_interval:
                angle = calculate_angle(previous_x_pred, previous_y_pred, predicted[0], predicted[1])
                capture_photo(frame, cx, cy, angle)
                last_photo_time = current_time

            if enable_sound and not is_sound_playing:
                # Play sound alert
                st.sidebar.audio(audio_bytes, format="audio/mp3", start_time=0)
                st.sidebar.write("🔊 Alert! Object moving down.")
                is_sound_playing = True
        elif previous_y_pred is not None and predicted[1] < previous_y_pred:
            is_sound_playing = False

        # Update previous positions
        previous_x_pred, previous_y_pred = predicted

        # Update graph data
        x_data_initial.append(cx)
        y_data_initial.append(cy)
        x_data_predicted.append(predicted[0])
        y_data_predicted.append(predicted[1])

        # Draw detected and predicted positions
        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)  # Initial position
        cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 10, (0, 0, 255), -1)  # Predicted position
        if previous_y_pred is not None:
            cv2.line(frame, (previous_x_pred, previous_y_pred), (cx, cy), (0, 255, 0), 2)

    # Display the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    # Update graph
    graph_placeholder.line_chart({"Initial": y_data_initial, "Predicted": y_data_predicted})

cap.release()
cv2.destroyAllWindows()
