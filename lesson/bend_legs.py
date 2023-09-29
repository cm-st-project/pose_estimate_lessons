import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 1)


# Initialize MediaPipe Pose solution
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open video capture (comment/uncomment appropriate line for webcam or video file)
# cap = cv2.VideoCapture(0)  # Use webcam
cap = cv2.VideoCapture('knee_angle.mp4')  # replace video.mp4 with your video
width = 800
height = 600
index = 0
frames = []
while True:
    # Read a frame from the video capture
    success, img = cap.read()

    if not success:
        break
    img = cv2.resize(img, (width, height))
    # Convert BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image with the pose estimation model
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        hip = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]
        ankle = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ANKLE]
        knee = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE]

        angle = (calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y), ))
        # print(angle)

        # append the frame when the person bend his knees
        if angle < 145:
            frames.append(index)
    index += 1
    cv2.waitKey(1)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(frames)
