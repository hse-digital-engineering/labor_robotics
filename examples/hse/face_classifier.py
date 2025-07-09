import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.utils.transform")

class FaceClassifier:
    def __init__(self, model_path="face_knn.joblib"):
        # Lade k-NN Modell
        self.knn = joblib.load(model_path)

        self.cap = None

        # Initialisiere InsightFace
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(960,640))  # GPU 0, fallback CPU

    def open_camera(self, cam_idx=0):
        # Starte Kamera
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Kamera konnte nicht geöffnet werden.")

    def get_predictions(self, frame):
        # Erkennt Gesichter in gegebenem Frame und klassifiziert diese

        results = []
        faces = self.app.get(frame)

        for face in faces:
            bbox = [int(v) for v in face.bbox]
            emb = face.embedding.reshape(1, -1)

            pred = self.knn.predict(emb)
            prob = self.knn.predict_proba(emb)
            max_prob = np.max(prob)

            results.append({
                "name": pred[0],
                "probability": max_prob,
                "bbox": bbox
            })

        return results
    
    @staticmethod
    def draw_bounding_box(frame, predictions):
        for pred in predictions:
            x1, y1, x2, y2 = pred["bbox"]
            label = f"{pred['name']} ({pred['probability'] * 100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def release(self):
        # Release camera
        self.cap.release()
        cv2.destroyAllWindows()
