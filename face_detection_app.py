import streamlit as st
import cv2
import tempfile
from detection_utils import detect_faces

st.set_page_config(page_title="Face Detection App", page_icon="🧠", layout="wide")
st.title("Face Detection App")
st.write("Detect faces from uploaded images. MTCNN is used by default.")

# Sidebar Controls
st.sidebar.header("⚙️ Settings")
method = st.sidebar.selectbox("Choose Detection Method", ["mtcnn", "haar"], index=0)  # MTCNN default

st.markdown("---")

# ------------------ Upload Image Mode ------------------
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
