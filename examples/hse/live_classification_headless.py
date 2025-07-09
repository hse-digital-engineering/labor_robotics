from face_classifier import FaceClassifier

fc = FaceClassifier()
fc.open_camera(0)

try:
    while True:
        # Frame + Klassifikationen holen
        ret, frame = fc.cap.read()
        if not ret:
            raise RuntimeError("⚠️ Kein Frame empfangen.")
        
        predictions = fc.get_predictions(frame)

        if predictions:
            print("Erkannte Gesichter:")
            for pred in predictions:
                print(f"- Name: {pred['name']}, "
                      f"Wahrscheinlichkeit: {pred['probability'] * 100:.1f}%, "
                      f"Bounding Box: {pred['bbox']}")
        else:
            print("⚠️ Kein Gesicht erkannt.")

except KeyboardInterrupt:
    print("\r  \rKeyboardInterrupt.")

finally:
    fc.release()
