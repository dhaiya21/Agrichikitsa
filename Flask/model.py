from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights

# Load ResNet34 with pretrained weights
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)

from PIL import Image
import io

app = Flask(__name__)

# Define your model class
class Crop_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)  # Change to a supported ResNet
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        out = self.network(xb)
        return out

# Load the model
model = Crop_Disease_Model()
model.load_state_dict(torch.load('./Models/crop_disease_detection.pth', map_location=torch.device('cpu')))
model.eval()

# List of class names
num_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
               'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
               'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
               'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Define the image transformation
transform = transforms.Compose(
    [transforms.Resize(size=128),
     transforms.ToTensor()])

# Prediction function
def predict_image(img):
    img_pil = Image.open(io.BytesIO(img))
    tensor = transform(img_pil)
    xb = tensor.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return num_classes[preds[0].item()]

# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Process image file
    img_bytes = file.read()
    prediction = predict_image(img_bytes)
    return jsonify({"prediction": prediction})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
