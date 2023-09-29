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
cap = cv2.VideoCapture('dance.mp4')  # replace video.mp4 with your video
width = 800
height = 600
index = 0
l_frames = []
r_frames = []
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
        if m <= -0.1:
            r_frames.append(index)
        elif m >= 0.1:  # If the value is 0.1 or higher, it indicates an imbalance to the left.:
            l_frames.append(index)
    index += 1
    cv2.waitKey(1)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print('left: ', l_frames)
print('right', r_frames)

"""
Output: 

left:  [40, 41, 42, 43, 44, 45, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 
 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 
 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252,
  268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 301, 302, 303, 304, 305, 
  306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 327, 365, 366, 367, 368, 369, 372, 373, 374, 375, 376, 
  377, 378, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457]
  
right [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
 62, 63, 64, 65, 66, 80, 81, 82, 83, 84, 92, 93, 94, 95, 96, 97, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 
 129, 130, 131, 132, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 258, 259, 260, 261, 262, 263, 293, 294, 
 295, 296, 297, 298, 345, 346, 347, 348, 349, 350, 384, 385, 386, 387, 388, 389, 390, 408, 409, 410, 411, 412, 413, 414, 
 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439]

"""
