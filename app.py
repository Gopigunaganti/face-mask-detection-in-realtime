from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
camera = cv2.VideoCapture(0)
model = load_model("model/mask_detector.h5")

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face = cv2.resize(frame, (224, 224)) / 255.0
            face = np.expand_dims(face, axis=0)
            pred = model.predict(face)[0]
            label = "Mask" if pred[0] > pred[1] else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
