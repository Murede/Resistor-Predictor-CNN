import torch
import torch.nn as nn
import torch.optim as optim
from model import ResistorCNN
from data_loader import get_data_loaders

def train_model(data_dir, num_classes, epochs=10, lr=0.001, batch_size=32, save_path='resistor_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    model = ResistorCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

if __name__ == "__main__":
    train_model(data_dir='data', num_classes=10)
