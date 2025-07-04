from flask import Flask, render_template, Response, jsonify, request, url_for
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import uuid

# Initialize Flask app
app = Flask(__name__)

# Load trained deep learning model
model = load_model('model.h5')

# Dictionary to interpret model predictions
class_dictionary = {0: 'no_car', 1: 'car'}

# Load video file
cap = cv2.VideoCapture('test/car_test.mp4')

# Load saved car parking positions
with open('car_pos.pkl', 'rb') as f:
    posList = pickle.load(f)

# Parking space dimensions (width x height)
width, height = 130, 65

# Upload folder for saving incoming user images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------
# Parking Space Detection Logic
# ------------------------------------
def checkParkingSpace(img):
    spaceCounter = 0
    imgCrops = []

    # Extract each parking region of interest (ROI)
    for pos in posList:
        x, y = pos
        imgCrop = img[y:y + height, x:x + width]
        imgResize = cv2.resize(imgCrop, (48, 48))          # Resize for model
        imgNormalized = imgResize / 255.0                  # Normalize
        imgCrops.append(imgNormalized)

    imgCrops = np.array(imgCrops)

    # Make predictions using the model
    predictions = model.predict(imgCrops)

    for i, pos in enumerate(posList):
        x, y = pos
        inID = np.argmax(predictions[i])
        label = class_dictionary[inID]

        # Draw bounding box and label
        if label == 'no_car':
            color = (0, 255, 0)       # Green = free
            thickness = 5
            spaceCounter += 1
            textColor = (0, 0, 0)
        else:
            color = (0, 0, 255)       # Red = occupied
            thickness = 2
            textColor = (255, 255, 255)

        # Draw rectangle around space
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

        # Add text label
        font_scale = 0.5
        text_thickness = 1
        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        textX = x
        textY = y + height - 5
        cv2.rectangle(img, (textX, textY - textSize[1] - 5), (textX + textSize[0] + 6, textY + 2), color, -1)
        cv2.putText(img, label, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, textColor, text_thickness)

    totalSpaces = len(posList)

    return img, spaceCounter, totalSpaces - spaceCounter

# ------------------------------------
# Video Frame Generator for Live Feed
# ------------------------------------
def generate_frames():
    while True:
        # Loop video when it ends
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (1280, 720))
        img, free_spaces, occupied_spaces = checkParkingSpace(img)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()

        # Yield frame as multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

# ------------------------------------
# Flask Routes
# ------------------------------------

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream video frame by frame
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get current count of free and occupied spaces
@app.route('/space_count')
def space_count():
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (1280, 720))
        _, free_spaces, occupied_spaces = checkParkingSpace(img)
        return jsonify(free=free_spaces, occupied=occupied_spaces)
    return jsonify(free=0, occupied=0)

# Route to upload and analyze a custom image
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file securely
    filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + "_" + filename
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    # Read uploaded image
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    img = cv2.resize(img, (1280, 720))

    # Analyze and save processed result
    processed_img, free_spaces, occupied_spaces = checkParkingSpace(img)
    processed_filename = "processed_" + unique_filename
    processed_filepath = os.path.join(UPLOAD_FOLDER, processed_filename)
    cv2.imwrite(processed_filepath, processed_img)

    # Generate URL for frontend to access image
    processed_image_url = url_for('static', filename='uploads/' + processed_filename)

    return jsonify({
        'processed_image_url': processed_image_url,
        'free_spaces': free_spaces,
        'occupied_spaces': occupied_spaces
    })

# ------------------------------------
# Run App
# ------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
