import cv2
import mediapipe as mp

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open video capture (comment/uncomment appropriate line for webcam or video file)
# cap = cv2.VideoCapture(0)  # Use webcam
cap = cv2.VideoCapture('hands_cross.mp4')  # replace video.mp4 with your video
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
    image_height, image_width, _ = img.shape
    # Convert BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image with the pose estimation model
    results = pose.process(imgRGB)

    if results.pose_landmarks:

        # Check the horizontal positions of the left and right wrists to identify frames where the left
        # wrist is to the left of the right wrist
        if (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width <=
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width):
            frames.append(index)
    index += 1
    cv2.waitKey(1)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(frames)

# output

"""
[28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 159,
 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 224, 378, 381, 382, 428, 431, 432, 433, 473, 474,
  475, 476, 477, 478, 479, 480]

"""
