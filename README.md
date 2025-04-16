# Real-time Face Mask Detection System

A real-time face mask detection system using OpenCV, TensorFlow, and Flask. This project detects whether a person is wearing a face mask or not in real-time using a webcam.

## Features

- Real-time face mask detection
- Web-based interface with live video feed
- Visual bounding boxes around detected faces
- Color-coded indicators (green for mask, red for no mask)
- Responsive design that works on all devices
- High accuracy detection using deep learning

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- Keras
- Flask
- NumPy
- Pillow

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/face-mask-detection-in-realtime.git
cd face-mask-detection-in-realtime
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained model and place it in the `model` directory:

- Download the model file from [here](link-to-model)
- Create a `model` directory if it doesn't exist
- Place the model file in the `model` directory

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open your web browser and go to:

```
http://localhost:5000
```

3. Allow camera access when prompted

4. The system will automatically detect faces and indicate whether a mask is being worn

## Project Structure

```
face-mask-detection-in-realtime/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main web interface
├── model/             # Pre-trained model directory
└── README.md          # Project documentation
```

## How It Works

1. The webcam captures video frames
2. Each frame is processed to detect faces using OpenCV
3. Detected faces are passed through a pre-trained deep learning model
4. The model predicts whether a mask is being worn
5. Results are displayed in real-time with visual indicators

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for face detection
- TensorFlow and Keras for deep learning
- Flask for web framework
