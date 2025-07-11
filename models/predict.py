import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = sorted(os.listdir("../data/EuroSAT"))

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_eurosat.pth", map_location=device))
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1459, 0.1132, 0.1137])
])

# todo: make more convenient format
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    predicted_class = classes[probs.argmax().item()]
    confidence = probs.max().item()

    return predicted_class, round(confidence, 3), probs.cpu().numpy()