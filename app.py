import torch
import cv2
import numpy as np
import streamlit as st

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load reference images for exercises
reference_images = {
    'Squat': 'correct_squat_image.jpg',
    'Push Up': 'correct_pushup_image.png',
    'Lunge': 'correct_lunge_image.png'
}

# Function to load and resize images
def load_and_resize(image_path, size=(320, 240)):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return image

# Load all reference images
loaded_images = {name: load_and_resize(path) for name, path in reference_images.items()}

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

        frame = cv2.flip(frame, 1)
        results = model(frame)

        for result in results.xyxy[0].numpy():
            if int(result[5]) == 0:  # Assuming 'person' class
                x1, y1, x2, y2 = map(int, result[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Pose detection logic remains here (using mediapipe or other methods)

        stframe.image(frame, channels='BGR')

    cap.release()
    cv2.destroyAllWindows()

st.title("Exercise Detection")
st.write("Ensure your webcam is enabled and select an exercise from the sidebar to begin detecting.")

exercise_option = st.sidebar.selectbox('Select an exercise:', ['Squat', 'Push Up', 'Lunge'])

if exercise_option in loaded_images:
    reference_image = loaded_images[exercise_option]
    if reference_image is not None:
        st.image(reference_image, caption=f'Correct {exercise_option} Position', use_column_width=True, channels='BGR')
    else:
        st.error(f"Failed to load the reference image for {exercise_option}. Path: {reference_images[exercise_option]}")

if st.button('Start'):
    detect_exercise(exercise_option)