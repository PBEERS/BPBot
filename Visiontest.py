import cv2
import serial
import time

# Initialize camera (Arducam IMX708)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize TF-Luna Lidar (UART, e.g., /dev/ttyS0)
try:
    lidar = serial.Serial('/dev/ttyS0', 115200, timeout=1)
except Exception as e:
    print(f"Failed to open Lidar serial: {e}")
    lidar = None

def read_lidar():
    if not lidar:
        return None
    while True:
        count = lidar.in_waiting
        if count >= 9:
            bytes_recv = lidar.read(9)
            if bytes_recv[0] == 0x59 and bytes_recv[1] == 0x59:
                distance = bytes_recv[2] + bytes_recv[3]*256
                return distance
        else:
            break
    return None

# Load a sample OpenCV model (e.g., face detector)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Vision model inference (face detection as example)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Read Lidar distance
    distance = read_lidar()
    if distance:
        cv2.putText(frame, f"Distance: {distance} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Vision + Lidar', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if lidar:
    lidar.close()
cv2.destroyAllWindows()