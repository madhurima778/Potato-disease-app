import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np

app = Flask(__name__)

# Load the ResNet feature extractor (already trained and saved)
resnet_model = models.resnet50()
resnet_model.fc = nn.Identity()  # Remove final classification layer
resnet_model.load_state_dict(torch.load('resnet_feature_extractor.pth', map_location='cpu'))
resnet_model.eval()

# Load the RandomForest model
rf_model = joblib.load('random_forest_model.pkl')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class names in your dataset
class_names = ['Early Blight','Late Blight','Healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file in the static folder
            image_path = os.path.join('static', file.filename)
            file.save(image_path)

            # Preprocess the image
            img = Image.open(image_path).convert('RGB')
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                features = resnet_model(img)

            features_np = features.view(features.size(0), -1).numpy()

            # Get prediction probabilities
            probs = rf_model.predict_proba(features_np)
            # Get the class with the highest probability (confidence score)
            confidence = np.max(probs) * 100
            confidence = round(confidence, 2)

            # Get the predicted class label
            prediction = rf_model.predict(features_np)
            predicted_class = class_names[prediction[0]]

            # Pass image_path (relative path) and prediction with confidence to the template
            return render_template('index.html', 
                                   prediction=predicted_class, 
                                   confidence=confidence, 
                                   image_path=file.filename)  # Pass only the file name

if __name__ == '__main__':
    app.run(debug=True)











