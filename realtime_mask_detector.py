import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the face mask detection model (.h5 format)
mask_model = load_model("model/mask_detector.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            # Preprocess the face for prediction
            face_resized = cv2.resize(face, (224, 224))  # Resize to model input size
            face_normalized = face_resized.astype("float") / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)

            # Predict mask or no mask
            pred = mask_model.predict(face_expanded)[0]
            (mask, withoutMask) = pred

            # Label and color based on prediction
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display label and rectangle
            cv2.putText(frame, f"{label}: {max(mask, withoutMask) * 100:.2f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        except Exception as e:
            print("Error during prediction:", e)

    # Show the output
    cv2.imshow("Face Mask Detector", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
