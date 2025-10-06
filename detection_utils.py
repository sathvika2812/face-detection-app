import cv2
import dlib
from mtcnn import MTCNN

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dlib_detector = dlib.get_frontal_face_detector()
mtcnn_detector = MTCNN()

def detect_faces(image, method="haar"):
    faces = []
    if method == "haar":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = haar_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in detections:
            faces.append((x, y, x + w, y + h))
    elif method == "dlib":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = dlib_detector(gray)
        for det in detections:
            faces.append((det.left(), det.top(), det.right(), det.bottom()))
    elif method == "mtcnn":
        detections = mtcnn_detector.detect_faces(image)
        for det in detections:
            x, y, w, h = det['box']
            faces.append((x, y, x + w, y + h))
    return faces
