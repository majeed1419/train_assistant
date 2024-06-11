import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Load reference images for exercises
reference_images = {
    'Squat': 'correct_squat_image.jpg',
    'Squat Front': 'squat_front.jpg',
    'Push Up': 'correct_pushup_image.png',
    'Lunge': 'correct_lunge_image.png'
}

# Function to load and resize images
def load_and_resize(image_path, size=(320, 240)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, size)
    return image

# Load all reference images
loaded_images = {name: load_and_resize(path) for name, path in reference_images.items()}

# Setup MediaPipe instance
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize counters and flags
correct_attempts = 0
exercise_started = False
correct_exercise = False

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_exercise(exercise):
    global correct_attempts, exercise_started, correct_exercise

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

            # Check correctness of exercise pose based on angles
            if exercise == 'Squat':
                keypoints = [mp_pose.PoseLandmark.LEFT_HIP,
                             mp_pose.PoseLandmark.LEFT_KNEE,
                             mp_pose.PoseLandmark.LEFT_ANKLE,
                             mp_pose.PoseLandmark.RIGHT_HIP,
                             mp_pose.PoseLandmark.RIGHT_KNEE,
                             mp_pose.PoseLandmark.RIGHT_ANKLE]
                threshold = (70, 110)

            elif exercise == 'Push Up':
                keypoints = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                             mp_pose.PoseLandmark.LEFT_ELBOW,
                             mp_pose.PoseLandmark.LEFT_WRIST,
                             mp_pose.PoseLandmark.RIGHT_SHOULDER,
                             mp_pose.PoseLandmark.RIGHT_ELBOW,
                             mp_pose.PoseLandmark.RIGHT_WRIST]
                threshold = (160, 180)

            elif exercise == 'Lunge':
                keypoints = [mp_pose.PoseLandmark.LEFT_HIP,
                             mp_pose.PoseLandmark.LEFT_KNEE,
                             mp_pose.PoseLandmark.LEFT_ANKLE,
                             mp_pose.PoseLandmark.RIGHT_HIP,
                             mp_pose.PoseLandmark.RIGHT_KNEE,
                             mp_pose.PoseLandmark.RIGHT_ANKLE]
                threshold = (70, 110)

            # Calculate angles and determine correctness
            angles = []
            for i in range(0, len(keypoints) - 2, 3):
                a = (landmarks[keypoints[i]].x, landmarks[keypoints[i]].y)
                b = (landmarks[keypoints[i + 1]].x, landmarks[keypoints[i + 1]].y)
                c = (landmarks[keypoints[i + 2]].x, landmarks[keypoints[i + 2]].y)
                angle = calculate_angle(a, b, c)
                angles.append(angle)

            correctness = "Incorrect"
            color = (0, 0, 255)  # Red color for incorrect
            if len(angles) == 2 and all(threshold[0] <= angle <= threshold[1] for angle in angles):  # Example thresholds for each exercise
                correctness = "Correct"
                color = (0, 255, 0)  # Green color for correct

            if correctness == "Correct":
                if not correct_exercise:
                    if not exercise_started:
                        exercise_started = True  # Start exercise detection
                    correct_attempts += 1  # Increment correct attempts
                    correct_exercise = True
            else:
                correct_exercise = False

        # Display correctness and attempts on the frame
        cv2.putText(frame, f"Status: {correctness}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Correct Attempts: {correct_attempts}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame in Streamlit
        stframe.image(frame, channels='BGR')

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

st.title("Exercise Detection")
st.write("Ensure your webcam is enabled and select an exercise from the sidebar to begin detecting.")

# Sidebar with exercise options
exercise_option = st.sidebar.selectbox('Select an exercise:', ['Squat', 'Push Up', 'Lunge'])

def load_and_resize(image_path, size=(320, 240)):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        # Resize the image with interpolation for better quality
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return image

# Load all reference images
loaded_images = {name: load_and_resize(path) for name, path in reference_images.items()}

# Display reference images for the selected exercise
if exercise_option in loaded_images:
    reference_image = loaded_images[exercise_option]
    if reference_image is not None:
        st.image(reference_image, caption=f'Correct {exercise_option} Position', use_column_width=True, channels='BGR')
    else:
        st.error(f"Failed to load the reference image for {exercise_option}. Path: {reference_images[exercise_option]}")


# Button to start the exercise detection
if st.button('Start'):
    detect_exercise(exercise_option)

# Display analysis of the exercise
# st.write(f"Total correct {exercise_option.lower()} attempts: {correct_attempts}")