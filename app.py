import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import os
import traceback
import json

app = Flask(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

try:
    # Load the pre-trained model
    print("Attempting to load model...")
    model_path = os.path.join('model', 'mask_detector.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Stack trace:")
    traceback.print_exc()
    model = None

# Load the face detector from OpenCV
try:
    print("Loading face detector...")
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"Face detector file not found at {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    print("Face detector loaded successfully")
except Exception as e:
    print(f"Error loading face detector: {e}")
    print("Stack trace:")
    traceback.print_exc()
    face_cascade = None

@app.route('/')
def index():
    return render_template('index.html')

def predict_mask(image):
    """Predict if the person is wearing a mask using the pre-trained model."""
    try:
        if model is None or face_cascade is None:
            raise Exception("Model or face detector not loaded properly")

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
            face_details.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            })

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
            try:
                predictions = model.predict(face_images, verbose=1)
                results = []

                for i, prediction in enumerate(predictions):
                    label = "Mask Detected" if prediction[0] > prediction[1] else "No Mask Detected"
                    results.append(label)

                print("Predictions:", results)
                return results, face_details
            except Exception as e:
                print(f"Error during model prediction: {e}")
                print("Stack trace:")
                traceback.print_exc()
                return ["Error in model prediction"], face_details

        print("No faces detected.")
        return ["No Face Detected"], face_details

    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return ["Error processing image"], []

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
            
        image = data['image']
        results, face_details = predict_mask(image)

        return jsonify({
            "label": results[0] if len(results) > 0 else "No Face Detected",
            "face_details": face_details
        })
    except Exception as e:
        print(f"Error in predict route: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        # Check if model and face detector are loaded
        if model is None or face_cascade is None:
            print("Error: Required models not loaded. Please check the model files.")
        else:
            # Run the app on localhost:5000
            print("Starting Flask server...")
            app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Stack trace:")
        traceback.print_exc()
