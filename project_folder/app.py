from flask import Flask, request, render_template, jsonify
import os
from utils import predict_crop_and_disease
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    crop, disease, confidence = predict_crop_and_disease(filepath)
    
    # Load crop info
    with open("crop_info.json") as f:
        crop_data = json.load(f)
    
    # Ensure the crop and disease are in the data
    crop_info = crop_data.get(crop, {})
    disease_info = crop_info.get(disease, {})

    
    return render_template('result.html', crop=crop, disease=disease, conf=confidence, info=disease_info)

if __name__ == '__main__':
    app.run(debug=True)
