import torch
from torchvision import transforms
from PIL import Image
from model import ResistorCNN

def predict(image_path, model_path='resistor_model.pth', num_classes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model = ResistorCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == "__main__":
    result = predict('sample_resistor.jpg')
    print(f'Predicted Class: {result}')
