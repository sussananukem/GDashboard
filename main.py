# Library imports
import streamlit as st
import warnings
import mediapipe as mp
import cv2
import numpy as np
from keras.models import load_model

warnings.filterwarnings('ignore')

# File imports
from data_loader import load_data, feature_names_mapping, categorical_keys, continuous_keys
from visualization import plot_feature_vs_birthweight, create_faceted_scatter_plot, plot_correlation_heatmap

# Set the page config
st.set_page_config(page_title="NeoWeight", layout="wide")

# Initialize gesture recognition model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = load_model('Model1/action.h5')
actions = ["comparison", "Grabbing", "HandRelation", "InspectObject", "MentalImage", "PlacementObject", "Pointing", "Rotation", "Separation"]
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Load the dataset
datafile = "data/birthwt_cleaned.csv"
df = load_data(datafile)

# Set up state management for zoom and alert
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 1  # Default zoom level

if 'alert_message' not in st.session_state:
    st.session_state.alert_message = None  # To store the latest alert message

# Custom CSS to make the video stream fixed at the top-right corner
st.markdown("""
    <style>
    .fixed-video {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 200px; /* Adjust the width */
        height: auto;
        z-index: 9999;
        border: 2px solid #f0f0f5;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for feature selection
st.sidebar.header("Feature Selection")
categorical_label = st.sidebar.selectbox(
    "Maternal Health Indicators",
    options=[feature_names_mapping[key] for key in categorical_keys],
    key='categorical'
)
continuous_label = st.sidebar.selectbox(
    "Pregnancy Development Factors",
    options=[feature_names_mapping[key] for key in continuous_keys],
    key='continuous'
)

categorical_name = [key for key, value in feature_names_mapping.items() if value == categorical_label][0]
continuous_name = [key for key, value in feature_names_mapping.items() if value == continuous_label][0]

# Placeholder for the plot (single instance to be updated)
plot_placeholder = st.empty()

# --- Function Definitions ---
def distance_between(p1_loc, p2_loc, landmarks):
    p1 = np.array([landmarks.landmark[p1_loc].x, landmarks.landmark[p1_loc].y, landmarks.landmark[p1_loc].z])
    p2 = np.array([landmarks.landmark[p2_loc].x, landmarks.landmark[p2_loc].y, landmarks.landmark[p2_loc].z])
    return np.linalg.norm(p1 - p2)

def landmark_to_dist_emb(landmarks):
    emb = np.array([
        distance_between(4, 8, landmarks),
        distance_between(4, 12, landmarks),
        distance_between(4, 16, landmarks),
        distance_between(4, 20, landmarks),
        distance_between(4, 0, landmarks),
        distance_between(8, 0, landmarks),
        distance_between(12, 0, landmarks),
        distance_between(16, 0, landmarks),
        distance_between(20, 0, landmarks),
        distance_between(8, 12, landmarks),
        distance_between(12, 16, landmarks),
        distance_between(1, 4, landmarks),
        distance_between(8, 5, landmarks),
        distance_between(12, 9, landmarks),
        distance_between(16, 13, landmarks),
        distance_between(20, 17, landmarks),
        distance_between(2, 8, landmarks),
        distance_between(2, 12, landmarks),
        distance_between(2, 16, landmarks),
        distance_between(2, 20, landmarks)
    ])
    emb_norm = emb / np.linalg.norm(emb)
    return emb_norm

# Function to get and adjust plot with zoom
def get_zoomed_plot(df, feature_key, zoom_level):
    fig = plot_feature_vs_birthweight(df, feature_key, "Birth Weight Category")
    # Adjust axis range based on zoom level
    fig.update_xaxes(range=[0, 10 / zoom_level])
    fig.update_yaxes(range=[0, 10 / zoom_level])
    return fig

# Display the initial plot with the current zoom level
plot_placeholder.plotly_chart(get_zoomed_plot(df, categorical_name, st.session_state.zoom_level), use_container_width=True)

# Create a placeholder for the video stream with the fixed class
video_placeholder = st.empty()

# Placeholder for alerts (ensure only one alert is displayed at a time)
alert_placeholder = st.empty()

# Buffer for storing frames
frames_buffer = []

# Open webcam and process frames
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.warning("Could not access the camera. Make sure it's connected.")
        break

    # Flip the frame horizontally for selfie-view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        left_landmarks = np.zeros(20)
        right_landmarks = np.zeros(20)
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == 'Left':
                left_landmarks = landmark_to_dist_emb(hand_landmarks)
            else:
                right_landmarks = landmark_to_dist_emb(hand_landmarks)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_features = np.concatenate((left_landmarks, right_landmarks))
        frames_buffer.append(frame_features)

        if len(frames_buffer) > 50:
            frames_buffer.pop(0)

    else:
        frames_buffer = []

    if len(frames_buffer) == 50:
        preprocessed_data = np.array(frames_buffer).reshape((1, 50, 40))
        prediction = model.predict(preprocessed_data)

        # Visualize prediction
        highest_prob_action_index = np.argmax(prediction)
        gesture_detected = actions[highest_prob_action_index]

        # Display one alert at a time (override the previous one)
        st.session_state.alert_message = f"Detected Gesture: {gesture_detected}"
        alert_placeholder.success(st.session_state.alert_message)

        # Perform action based on gesture
        if gesture_detected == "Pointing" or gesture_detected == "Separation" or gesture_detected == "Grabbing" or gesture_detected == "InspectObject" or gesture_detected == "MentalImage":
            st.session_state.zoom_level = min(st.session_state.zoom_level + 0.5, 10)  # Zoom in (max zoom level 10)
            alert_placeholder.success(f"Zooming in ({gesture_detected} detected)")
        elif gesture_detected == "HandRelation" or gesture_detected == "comparison" or gesture_detected == "Rotation" or gesture_detected == "PlacementObject":
            st.session_state.zoom_level = max(st.session_state.zoom_level - 0.5, 0.1)  # Zoom out (min zoom level 0.1)
            alert_placeholder.warning(f"Zooming out ({gesture_detected} detected)")

        # Display the gesture detected on the frame
        cv2.putText(frame, f"Action: {gesture_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Update the existing plot based on the new zoom level
        plot_placeholder.plotly_chart(get_zoomed_plot(df, categorical_name, st.session_state.zoom_level), use_container_width=True)

    # Display the frame inside the Streamlit app (with fixed-video class)
    video_placeholder.image(frame, channels='BGR', use_column_width=False, width=300)

    # Allow exit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
