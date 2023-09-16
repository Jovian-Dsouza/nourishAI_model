import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import uuid
import shutil

app = Flask(__name__)
app.config['MODEL_PATH'] = os.environ.get('MODEL_PATH', 'mobilenet_v3_large_final.h5')
app.config['LABEL_PATH'] = os.environ.get('LABEL_PATH', 'labels.txt')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create the directory if it doesn't exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
labels = None

def load_model_and_labels():
    global model, labels
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    with open(app.config['LABEL_PATH'], 'r') as f:
        labels = [label.strip() for label in f.readlines()]

def preprocess_image(image_path):
    """Preprocess the input image for MobileNet V3 model."""
    with Image.open(image_path) as image:
        image = image.resize((224, 224))
        image_array = np.asarray(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

@app.route('/')
def index():
    """Render the main page."""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for prediction. Accepts an image and returns the prediction label as JSON."""
    image_file = request.files.get('image')
    
    if not image_file or image_file.filename == '':
        return jsonify(error="No image provided"), 400
    
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_file.save(temp_filename)
    
    try:
        preprocessed_image = preprocess_image(temp_filename)
        predictions = model.predict(preprocessed_image)
        return jsonify({'predictions': labels[np.argmax(predictions[0])]})
    finally:
        os.remove(temp_filename)

@app.route('/predict_page', methods=['POST'])
def predict():
    """Web endpoint for prediction. Accepts an image and renders the results on the web page."""
    image_file = request.files.get('image')
    
    if not image_file or image_file.filename == '':
        return redirect(request.url)

    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_file.save(temp_filename)

    # Save the image to the upload folder
    shutil.copy(temp_filename, os.path.join(app.config['UPLOAD_FOLDER'], temp_filename))
    
    try:
        preprocessed_image = preprocess_image(temp_filename)
        predictions = model.predict(preprocessed_image)
        return render_template('upload.html', prediction=labels[np.argmax(predictions[0])], image_url=url_for('uploaded_file', filename=temp_filename))
    finally:
        os.remove(temp_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    load_model_and_labels()
    app.run(debug=True)
