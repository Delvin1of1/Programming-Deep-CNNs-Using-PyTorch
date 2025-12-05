import Utils
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Resnet50Model import Bottleneck, ResNet, ResNet50


def main():
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    net = ResNet50(10).to(device)
    
    # CIFAR-10 class names
    classes = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Load data
    _, trainloader, _, testloader = Utils.get_loaders()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=5
    )
    
    # Training parameters
    EPOCHS = 200
    
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # Track losses
            losses.append(loss.item())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Running loss for printing
            running_loss += loss.item()
            
            # Print progress every 100 batches
            if i % 100 == 0 and i > 0:
                avg_loss = running_loss / 100
                print(f'Loss [Epoch {epoch+1}, Batch {i}]: {avg_loss:.4f}')
                running_loss = 0.0
        
        # Adjust learning rate based on average loss
        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}')
    
    print('\n--- Training Done ---')
    
    # Compute accuracy on test set
    print('\n--- Computing Accuracy on Test Set ---')
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Track accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * (correct / total)
    print(f'Accuracy on 10,000 test images: {accuracy:.2f}%')


if __name__ == "__main__":
    sys.exit(int(main() or 0))
