import cv2
import mediapipe as mp


def calculate_slope(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m


# Initialize MediaPipe Pose solution
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open video capture (comment/uncomment appropriate line for webcam or video file)
# cap = cv2.VideoCapture(0)  # Use webcam
cap = cv2.VideoCapture('raise_hands.mp4')  # replace video.mp4 with your video
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
        l_shoulder = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]

        m = (calculate_slope(r_shoulder.x, r_shoulder.y, l_shoulder.x, l_shoulder.y, ))
        # print(m)
        # If the value is -0.1 or lower, it indicates an imbalance to the right in the shoulder.
        # If the value is 0.1 or higher, it indicates an imbalance to the left.
        if -0.1 < m < 0.1:  #

            frames.append(index)
    index += 1
    cv2.waitKey(1)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(frames)
