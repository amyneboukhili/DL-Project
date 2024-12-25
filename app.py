from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model('chest_xray_cnn_model.h5')


img_size = (150, 150)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=img_size) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
   
    img_array = preprocess_image(file_path)
    
  
    prediction = model.predict(img_array)
    
   
    result = 'Pneumonia' if prediction[0] > 0.5 else 'Normal'
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
