import streamlit as st
import cv2
import tempfile
import numpy as np
from detection_utils import detect_faces

st.set_page_config(page_title="Face Detection App", page_icon="🧠", layout="wide")
st.title("Face Detection App")
st.write("Detect faces in real-time or from uploaded images. MTCNN is used by default.")

# Sidebar Controls
st.sidebar.header("⚙️ Settings")
method = st.sidebar.selectbox("Choose Detection Method", ["mtcnn", "haar"], index=0)  # MTCNN default
mode = st.sidebar.radio("Choose Mode", ["📷 Webcam", "🖼️ Upload Image"])

st.markdown("---")

# ------------------ Webcam Mode ------------------
if mode == "📷 Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    st.markdown("**Tip:** Uncheck the box to stop the webcam.")

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        faces = detect_faces(frame, method)

        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, method.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
    st.info("Webcam stopped.")

# ------------------ Upload Image Mode ------------------
elif mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        image = cv2.imread(tfile.name)
        faces = detect_faces(image, method)

        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, method.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption=f"Detected {len(faces)} face(s) using {method.upper()}",
                 use_column_width=True)
