import math

import cv2
import mediapipe as mp


def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    if angle < 0:
        angle += 360
    angle = min(angle, 360 - angle)

    return round(angle, 1)


# Initialize MediaPipe Pose solution
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open video capture (comment/uncomment appropriate line for webcam or video file)
# cap = cv2.VideoCapture(0)  # Use webcam
cap = cv2.VideoCapture('downward-dog.mp4')  # replace video.mp4 with your video
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

        elbow = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW]
        shoulder = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]
        wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]

        elbow_angle = (calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y)))

        print(elbow_angle)

        # append the frame when the person bend his knees
        if round(elbow_angle) >= 165:
            frames.append(index)
    index += 1
    cv2.waitKey(1)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(frames)

# output
""""
[0, 1, 2, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 
114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 
138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 
162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 
186, 187, 188, 189, 190]

"""
