from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/detection-methods.html')
def detection_methods():
    return render_template('detection-methods.html')

@app.route('/common-diseases.html')
def common_diseases():
    return render_template('common-diseases.html')

@app.route('/treatment-prevention.html')
def treatment_prevention():
    return render_template('treatment-prevention.html')

@app.route('/additional-resources.html')
def additional_resources():
    return render_template('additional-resources.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = Image.open(file_path)
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Model definition and prediction
        model = tf.keras.Sequential([
            tf.keras.applications.DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(256, 256, 3),
                pooling='avg'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        prediction = model.predict(image_array)
        probability = float(prediction[0][0])  # Convert to native Python float
        print(f"Prediction probability: {probability}")

        return jsonify({"probability": probability})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/detection-methods.html')
def detection_methods():
    return render_template('detection-methods.html')

@app.route('/common-diseases.html')
def common_diseases():
    return render_template('common-diseases.html')

@app.route('/treatment-prevention.html')
def treatment_prevention():
    return render_template('treatment-prevention.html')

@app.route('/additional-resources.html')
def additional_resources():
    return render_template('additional-resources.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = Image.open(file_path)
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Model definition and prediction
        model = tf.keras.Sequential([
            tf.keras.applications.DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(256, 256, 3),
                pooling='avg'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        prediction = model.predict(image_array)
        probability = float(prediction[0][0])  # Convert to native Python float
        print(f"Prediction probability: {probability}")

        return jsonify({"probability": probability})

if __name__ == '__main__':
    app.run(debug=True)
