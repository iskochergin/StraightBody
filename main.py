import cv2
from mediapipe.python.solutions import pose
from mediapipe.python.solutions.pose import PoseLandmark
import numpy as np
from collections import deque
from plyer import notification
import time
import argparse

parser = argparse.ArgumentParser(description="Body Position Detection with Notifications")
parser.add_argument('--show_camera', action='store_true', help='Display the camera feed window')
args = parser.parse_args()

EPS_DISTANCE = 15
PROCESS_EVERY_NTH_FRAME = 5
distance_history = deque(maxlen=10)
alignment_history = deque(maxlen=10)
NOTIFICATION_COOLDOWN = 10
last_notification_time = 0


def calculate_shoulder_distance(left, right):
    return np.hypot(left[0] - right[0], left[1] - right[1])


def is_body_aligned(left, right, eps):
    return abs(left[1] - right[1]) <= eps


def is_body_straight(distances, eps):
    if len(distances) < 2:
        return True
    avg_distance = np.mean(distances)
    return abs(distances[-1] - avg_distance) <= eps


def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        app_name='Body Position Detector',
        timeout=5
    )


with pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose_detector:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_NTH_FRAME != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(frame_rgb)
        problem_detected = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            left_coords = (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0]))
            right_coords = (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0]))
            shoulder_distance = calculate_shoulder_distance(left_coords, right_coords)
            aligned = is_body_aligned(left_coords, right_coords, EPS_DISTANCE)
            distance_history.append(shoulder_distance)
            alignment_history.append(aligned)
            straight = is_body_straight(distance_history, EPS_DISTANCE)
            problem_detected = not (aligned and straight)

            if problem_detected:
                current_time = time.time()
                if current_time - last_notification_time > NOTIFICATION_COOLDOWN:
                    send_notification(
                        title="Posture Alert",
                        message="Your shoulders are not aligned or you're leaning forward."
                    )
                    last_notification_time = current_time

            if args.show_camera:
                color = (0, 255, 0) if not problem_detected else (0, 0, 255)
                cv2.line(frame, left_coords, right_coords, color, 2)
                cv2.circle(frame, left_coords, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_coords, 5, (255, 0, 0), -1)
                status_text = "Aligned and Straight" if not problem_detected else "Not Aligned or Straight"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if args.show_camera:
            cv2.imshow("Body Position Detection", frame)

        if args.show_camera and cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    if args.show_camera:
        cv2.destroyAllWindows()
