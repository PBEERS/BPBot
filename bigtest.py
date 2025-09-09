import cv2
import time
import numpy as np
from tf_luna import TfLuna
from picamera2 import Picamera2

import RPi.GPIO as GPIO

# Servo setup
SERVO_X_PIN = 18
SERVO_Y_PIN = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_X_PIN, GPIO.OUT)
GPIO.setup(SERVO_Y_PIN, GPIO.OUT)
servo_x = GPIO.PWM(SERVO_X_PIN, 50)
servo_y = GPIO.PWM(SERVO_Y_PIN, 50)
servo_x.start(7.5)  # Center position
servo_y.start(7.5)

# Lidar setup
lidar = TfLuna('/dev/ttyUSB0')  # Adjust port as needed
#I believe this is ttySERIAL0 for us, maybe ttyserial0 or ttyser0

# Camera setup
picam2 = Picamera2()
picam2.start()
time.sleep(2)

# Function to set servo angle
def set_servo_angle(servo, angle):
    duty = 2.5 + (angle / 18.0)
    servo.ChangeDutyCycle(duty)

# Function to find red cup in frame
def find_red_cup(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2
            return (cx, cy), (x, y, w, h)
    return None, None

try:
    x_angle = 90
    y_angle = 90
    frame_width = 640
    frame_height = 480
    while True:
        frame = picam2.capture_array()
        frame = cv2.resize(frame, (frame_width, frame_height))
        center, bbox = find_red_cup(frame)
        if center:
            cx, cy = center
            error_x = cx - frame_width // 2
            error_y = cy - frame_height // 2
            x_angle -= error_x * 0.05
            y_angle += error_y * 0.05
            x_angle = max(0, min(180, x_angle))
            y_angle = max(0, min(180, y_angle))
            set_servo_angle(servo_x, x_angle)
            set_servo_angle(servo_y, y_angle)
            cv2.rectangle(frame, bbox[:2], (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)
            cv2.circle(frame, center, 5, (255,0,0), -1)
            distance = lidar.read_distance()
            cv2.putText(frame, f"Distance: {distance}cm", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    servo_x.stop()
    servo_y.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()