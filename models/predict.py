import torch
import torchvision.transforms as v2
from torchvision import models
from PIL import Image

from app.config import settings_factory

settings = settings_factory()

SPACE_CLASSES = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial', 'Pasture',
                 'Permanent Crop', 'Residential', 'River', 'Sea Lake']
MODEL_PATH = settings.MODEL_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# todo: more generic way
def load_model():
    resnet18_model = models.resnet18()
    resnet18_model.fc = torch.nn.Linear(resnet18_model.fc.in_features, 10)
    resnet18_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    resnet18_model = resnet18_model.to(device)

    resnet18_model.eval()

    return resnet18_model

transform = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1459, 0.1132, 0.1137])
])

# todo: make more convenient format
def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    predicted_class = SPACE_CLASSES[probs.argmax().item()]
    confidence = probs.max().item()

    return predicted_class, round(confidence, 3), probs.cpu().numpy()