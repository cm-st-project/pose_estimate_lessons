import cv2
import mediapipe as mp

# For static images:
mp_pose = mp.solutions.pose
file = 'raise_hands.jpg'  # Replace the raise_hands.jpg with your image
BG_COLOR = (192, 192, 192)  # gray
with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # the coordinates of the left wrist and shoulder landmark in the format: "coordinates: (x, y)"
    # where x is the horizontal position (scaled by image width) and y is the vertical position (scaled by image height)
    # x and y: These landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
    print(
        f'Left wrist coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height})'
    )
    print(
        f'Left wrist coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height})'
    )

    if (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height >
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height):
        print('The left hand is above the shoulder')
    else:
        print('The left hand is below the shoulder')

#################################################################################################
# Video

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


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
    image_height, image_width, _ = img.shape
    # Convert BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image with the pose estimation model
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        # Check if the left wrist is above the left shoulder
        if (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height >
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height):
            frames.append(index)
    index += 1
    cv2.waitKey(1)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(frames)
