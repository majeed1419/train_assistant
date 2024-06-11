import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.write("OpenCV Version:", cv2.__version__)
# Load reference images for correct squat position
reference_image_side_path = 'correct_squat_image.jpg'
reference_image_front_path = 'squat_front.jpg'
reference_image_side = cv2.imread(reference_image_side_path)
reference_image_front = cv2.imread(reference_image_front_path)
reference_image_side = cv2.resize(reference_image_side, (320, 240))  # Resize for display
reference_image_front = cv2.resize(reference_image_front, (320, 240))  # Resize for display

# Setup MediaPipe instance
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define key points for squat pose
squat_keypoints = [mp_pose.PoseLandmark.LEFT_HIP,
                   mp_pose.PoseLandmark.LEFT_KNEE,
                   mp_pose.PoseLandmark.LEFT_ANKLE,
                   mp_pose.PoseLandmark.RIGHT_HIP,
                   mp_pose.PoseLandmark.RIGHT_KNEE,
                   mp_pose.PoseLandmark.RIGHT_ANKLE]

# Initialize counters and flags
correct_squat_attempts = 0
squat_started = False
correct_squat = False

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_squat():
    global correct_squat_attempts, squat_started, correct_squat

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and get pose landmarks
        results = pose.process(image_rgb)

        # Draw the pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get the landmark positions
            landmarks = results.pose_landmarks.landmark

            # Check correctness of squat pose based on angles
            angles = []
            for i in range(0, len(squat_keypoints) - 2, 3):
                a = (landmarks[squat_keypoints[i]].x, landmarks[squat_keypoints[i]].y)
                b = (landmarks[squat_keypoints[i + 1]].x, landmarks[squat_keypoints[i + 1]].y)
                c = (landmarks[squat_keypoints[i + 2]].x, landmarks[squat_keypoints[i + 2]].y)
                angle = calculate_angle(a, b, c)
                angles.append(angle)

            correctness = "Incorrect"
            color = (0, 0, 255)
            if len(angles) == 2 and all(70 <= angle <= 110 for angle in angles):  # Example thresholds for squat
                correctness = "Correct"
                color = (0, 255, 0)

            if correctness == "Correct":
                if not correct_squat:
                    if not squat_started:
                        squat_started = True  # Start squat detection
                    correct_squat_attempts += 1  # Increment correct squat attempts
                    correct_squat = True
            else:
                correct_squat = False

        # Display correctness and attempts on the frame
        cv2.putText(frame, f"Status: {correctness}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Correct Squat Attempts: {correct_squat_attempts}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the frame in Streamlit
        stframe.image(frame, channels='BGR')

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

st.title("Squat Exercise Detection")
st.write("Ensure your webcam is enabled and click the 'Start' button to begin detecting squat exercises.")

# Display reference images for correct squat position
st.image(reference_image_side, caption='Side View of Correct Squat Position', use_column_width=True)
st.image(reference_image_front, caption='Front View of Correct Squat Position', use_column_width=True)

# Button to start the squat detection
if st.button('Start'):
    detect_squat()
