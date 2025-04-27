from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the pre-trained ResNet model with modified output layer for 5 classes (adjust based on your use case)
def load_model():
    model = models.resnet50(weights='IMAGENET1K_V1')  # Load pre-trained ResNet50 model
    # Modify the fully connected layer to match your class size (5 instead of 1000)
    model.fc = nn.Linear(model.fc.in_features, 5)  # 5 output classes, modify as needed

    # Load the pre-trained weights, but exclude the fully connected layer
    state_dict = torch.load('model/model.pth', map_location=torch.device('cpu'))

    # Remove the 'fc' layer from the loaded state dict to avoid the size mismatch error
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)

    # Load the weights into the model, excluding the 'fc' layer
    model.load_state_dict(state_dict, strict=False)

    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.Lambda(lambda img: img.convert('RGB')),  # Convert to RGB if not already
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image and apply the transformation
        image = Image.open(io.BytesIO(file.read()))
        
        # Apply the transformation to ensure correct format and size
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)
        
        # Placeholder for caption generation logic
        # Replace this with your actual caption logic
        captions = ["A description of the image", "Another caption", "More possible captions"]
        predicted_caption = captions[predicted_class.item()]  # Use the predicted class as an index
        
        return jsonify({'caption': predicted_caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
