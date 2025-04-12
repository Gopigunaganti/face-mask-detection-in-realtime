import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model/mask_detector.h5')

# Load the face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

def predict_mask(image):
    """Predict if the person is wearing a mask using the pre-trained model."""
    try:
        # Convert base64 image to OpenCV format
        print("Processing image...")
        img_data = base64.b64decode(image.split(',')[1])  # Remove 'data:image/jpeg;base64,' part
        img = Image.open(BytesIO(img_data))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        print("Image converted to BGR format")

        # Detect faces in the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        print(f"Detected {len(faces)} face(s)")

        # Prepare the response
        face_details = []
        for (x, y, w, h) in faces:
            face_details.append({"x": x, "y": y, "width": w, "height": h})

        # If faces are found, predict mask for each face
        if len(faces) > 0:
            face_images = []
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                face = face.astype("float32") / 255.0
                face_images.append(face)

            face_images = np.array(face_images)

            # Check if the model is working correctly
            print("Making predictions...")
            predictions = model.predict(face_images)
            results = []

            for i, prediction in enumerate(predictions):
                label = "Mask Detected" if prediction[0] > prediction[1] else "No Mask Detected"
                results.append(label)

            print("Predictions:", results)
            return results, face_details

        print("No faces detected.")
        return ["No Face Detected"], face_details

    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["Error processing image"], []

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = data['image']
    results, face_details = predict_mask(image)

    # Ensure that the results are serializable and return a proper JSON response
    return jsonify({
        "label": results[0] if len(results) > 0 else "No Face Detected",
        "face_details": face_details
    })

if __name__ == "__main__":
    app.run(debug=True)
