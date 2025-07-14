import torch
from PIL.Image import Image

from models.predict import SPACE_CLASSES, load_model, device, transform

model = load_model()

class AIService:

    @staticmethod
    def predict(img: Image):
        image = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = SPACE_CLASSES[probs.argmax().item()]
            confidence = probs.max().item()

        return {
            "main_class": predicted_class,
            "main_probability": round(confidence, 3),
            "other_classes": [{ "class": SPACE_CLASSES[idx], "probability": round(probs[idx].item(), 3)} for idx in range(len(probs))]
        }
