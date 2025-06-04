import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        pass

    @staticmethod
    def detect_faces(frame: np.ndarray):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Alternatives:
        # haarcascade_frontalcatface.xml
        # haarcascade_frontalface_alt2.xml
        # haarcascade_frontalface_alt_tree.xml
        # haarcascade_frontalface_alt.xml

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=11)
        return faces
    
    @staticmethod
    def draw_bounding_box(frame: np.ndarray, faces):
        rectangles = []

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, frame.shape[0] - 21), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for (x, y, w, h) in faces:
                top_left = (x, y)
                top_right = (x + w, y)
                bottom_right = (x + w, y + h)
                bottom_left = (x, y + h)
                rectangles.append([top_left, top_right, bottom_right, bottom_left])
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        return frame