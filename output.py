import cv2
import numpy as np
import pygame
import math
from datetime import datetime
from flask import Flask, render_template, jsonify
from kalmanfilter import KalmanFilter
from orange_detector import OrangeDetector  # Replace with your object detector import

app = Flask(__name__)

# Initialize Kalman Filter and Orange Detector
kf = KalmanFilter()
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

# Initialize Flask variables for sending data
downward_direction = False
coordinates = {'x': 0, 'y': 0}

# Function to calculate the angle between two points
def calculate_angle(prev_x, prev_y, curr_x, curr_y):
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    angle = math.degrees(math.atan2(dy, dx))  # Calculate angle in degrees
    return angle

# Function to update the object position and check for downward movement
def update_position(frame):
    global previous_x_pred, previous_y_pred, is_sound_playing, last_photo_time, downward_direction, coordinates

    # Detect object (orange in this case)
    orange_bbox = od.detect(frame)
    if orange_bbox is None:
        return

    x, y, x2, y2 = orange_bbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    # Predict next position using Kalman filter
    predicted = kf.predict(cx, cy)

    current_time = datetime.now().timestamp()
    if previous_y_pred is not None and predicted[1] > previous_y_pred:
        downward_direction = True  # Object is moving down
        coordinates['x'], coordinates['y'] = predicted[0], predicted[1]
        if current_time - last_photo_time >= photo_interval:
            angle = calculate_angle(previous_x_pred, previous_y_pred, predicted[0], predicted[1])
            last_photo_time = current_time  # Update last photo time

        # Play sound when moving down
        if not is_sound_playing:
            sound.play()
            is_sound_playing = True
    elif previous_y_pred is not None and predicted[1] < previous_y_pred:
        downward_direction = False  # Object is moving up
        if is_sound_playing:
            pygame.mixer.stop()
            is_sound_playing = False

    # Update previous positions
    previous_x_pred = predicted[0]
    previous_y_pred = predicted[1]

@app.route('/get_coordinates')
def get_coordinates():
    return jsonify(coordinates=coordinates, downward_direction=downward_direction)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
