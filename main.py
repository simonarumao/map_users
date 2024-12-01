import cv2
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import math
import requests  # For triggering Flask server
from kalmanfilter import KalmanFilter
from orange_detector import OrangeDetector  # Replace with your object detector import

# Flask server URL
FLASK_SERVER_URL = "http://127.0.0.1:5000/trigger"

cap = cv2.VideoCapture('ball.mp4')  # Use your video file here

# Load Kalman filter
kf = KalmanFilter()

# Load Orange Detector
od = OrangeDetector()

# Initialize pygame for sound
pygame.mixer.init()
sound = pygame.mixer.Sound("alert.mp3")  # Replace with the actual sound file path

# Initialize variables
previous_x_pred = None
previous_y_pred = None
is_sound_playing = False
last_photo_time = 0
photo_interval = 2  # Time in seconds between capturing photos

# Initialize graph data
x_data_initial = []
y_data_initial = []
x_data_predicted = []
y_data_predicted = []

# Function to calculate the angle between two points
def calculate_angle(prev_x, prev_y, curr_x, curr_y):
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    angle = math.degrees(math.atan2(dy, dx))  # Calculate angle in degrees
    return angle

# Function to send trigger to Flask server
def send_trigger(lat, lng):
    try:
        response = requests.post(FLASK_SERVER_URL, json={"latitude": lat, "longitude": lng})
        print(response.json())
    except Exception as e:
        print("Error sending trigger:", e)

# Function to capture a photo with the angle text
def capture_photo(frame, cx, cy, angle):
    font = cv2.FONT_HERSHEY_SIMPLEX
    angle_text = f"Angle: {angle:.2f}°"
    cv2.putText(frame, angle_text, (cx - 50, cy - 20), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the image with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"photo_{timestamp}.png"
    cv2.imwrite(filename, frame)
    print(f"Captured photo: {filename}")

# Live update function
def update(frame):
    global previous_x_pred, previous_y_pred, is_sound_playing, last_photo_time

    ret, frame = cap.read()
    if not ret:
        ani.event_source.stop()

    # Detect object (orange in this case)
    orange_bbox = od.detect(frame)
    if orange_bbox is None:
        return line_initial, line_predicted  # Skip if no object is detected

    x, y, x2, y2 = orange_bbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    # Predict next position using Kalman filter
    predicted = kf.predict(cx, cy)

    # Check if object is moving down and capture photo if necessary
    current_time = datetime.now().timestamp()
    if previous_y_pred is not None and predicted[1] > previous_y_pred:
        # Trigger Flask alert with live GPS location
        live_lat = 19.0760  # Replace with actual latitude
        live_lng = 72.8777  # Replace with actual longitude
        send_trigger(live_lat, live_lng)

        if current_time - last_photo_time >= photo_interval:
            angle = calculate_angle(previous_x_pred, previous_y_pred, predicted[0], predicted[1])
            capture_photo(frame, cx, cy, angle)
            last_photo_time = current_time  # Update last photo time

        # Play sound when moving down
        if not is_sound_playing:
            sound.play()
            is_sound_playing = True
    elif previous_y_pred is not None and predicted[1] < previous_y_pred:
        if is_sound_playing:
            pygame.mixer.stop()
            is_sound_playing = False

    # Update previous positions
    previous_x_pred = predicted[0]
    previous_y_pred = predicted[1]

    # Update graph data
    x_data_initial.append(cx)
    y_data_initial.append(cy)
    x_data_predicted.append(predicted[0])
    y_data_predicted.append(predicted[1])

    # Draw initial (blue) and predicted (red) positions
    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)  # Blue dot for initial position
    cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 10, (0, 0, 255), -1)  # Red dot for predicted position

    # Draw lines and angle
    if previous_y_pred is not None:
        # Draw green line between previous and current position
        cv2.line(frame, (previous_x_pred, previous_y_pred), (cx, cy), (0, 255, 0), 2)

        # Draw predicted position
        cv2.line(frame, (cx, cy), (int(predicted[0]), int(predicted[1])), (0, 255, 0), 2)

        # Calculate and display the angle
        angle = calculate_angle(previous_x_pred, previous_y_pred, cx, cy)
        angle_text = f"Angle: {angle:.2f}°"
        cv2.putText(frame, angle_text, (int(predicted[0]) + 10, int(predicted[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Update previous positions
    previous_x_pred = predicted[0]
    previous_y_pred = predicted[1]

    # Convert OpenCV frame to RGB for matplotlib display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax1.imshow(frame_rgb)  # Update the object detection window

    # Update the graph
    line_initial.set_data(x_data_initial, y_data_initial)
    line_predicted.set_data(x_data_predicted, y_data_predicted)

    # Show the updated graph
    return line_initial, line_predicted

# Initialize the figure for plotting the graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_title("Object Detection")
ax1.axis('off')  # Hide axis for object detection frame

# Initialize plot for positions
ax2.set_xlim(0, 640)
ax2.set_ylim(0, 480)
ax2.set_title("Position Graph")
ax2.set_xlabel("X Position")
ax2.set_ylabel("Y Position")

# Create plot lines for initial and predicted positions
line_initial, = ax2.plot([], [], marker='o', linestyle='-', color='b', markersize=10, label='Initial Position')
line_predicted, = ax2.plot([], [], marker='o', linestyle='-', color='r', markersize=10, label='Predicted Position')
ax2.legend()

# Run the animation
ani = FuncAnimation(fig, update, blit=False, interval=5)

# Show the matplotlib plot (graph window)
plt.show()

cap.release()
cv2.destroyAllWindows()
