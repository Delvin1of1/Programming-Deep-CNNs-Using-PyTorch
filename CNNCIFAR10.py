import sys
import Utils
from CNNModel import CNNModel
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters
    path_data = './data/'
    batch_size = 16
    num_epochs = 100
    
    # Load data
    trainds, trainloader, testds, testloader = Utils.prepare_data(path_data, batch_size)
    
    # Explore dataset
    train_iter = iter(trainloader)
    images, labels = next(train_iter)  # Get a batch (e.g., 16x3x32x32)
    print(f"Image shape: {images[0].shape}")
    
    # Class mappings
    d_class2idx = trainds.class_to_idx  # Class names to indices
    print(f"Class to index: {d_class2idx}")
    
    d_idx2class = dict(zip(d_class2idx.values(), d_class2idx.keys()))
    print(f"Index to class: {d_idx2class}")
    
    # Visualize training images
    images, labels = next(train_iter)
    Utils.plot_images(images, labels)
    print(' '.join('%5s' % d_idx2class[int(labels[j])] for j in range(len(images))))
    plt.show()
    
    # Initialize model
    net = CNNModel()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # Training loop
    trainloader = torch.utils.data.DataLoader(
        trainds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    running_loss = 0
    printfreq = 1000
    
    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            loss = loss_criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
            
            if i % printfreq == printfreq - 1:
                print(f'Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss/printfreq:.4f}')
                running_loss = 0
    
    # Print model state
    print("\n--- Model Parameters ---")
    for param_tensor in net.state_dict():
        print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")
    
    print("\n--- Optimizer State ---")
    print(f"Optimizer keys: {optimizer.state_dict().keys()}")
    print(f"Param groups: {optimizer.state_dict()['param_groups']}")
    
    # Save model
    fname = './models/CIFAR10_cnn.pth'
    torch.save(net.state_dict(), fname)
    print(f"\nModel saved to {fname}")
    
    # Load saved model
    loaded_dict = torch.load(fname)
    net.load_state_dict(loaded_dict)
    net.eval()
    
    # Test model predictions
    print("\n--- Actual Test Images ---")
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    Utils.plot_images(images, labels)
    print(' '.join('%5s' % d_idx2class[int(labels[j])] for j in range(len(images))))
    plt.show()
    
    print("\n--- Predictions ---")
    preds = net(images)
    preds = preds.argmax(dim=1)
    Utils.plot_images(images, preds)
    print(' '.join('%5s' % d_idx2class[int(preds[j])] for j in range(len(images))))
    plt.show()
    
    # Compute accuracy on test set
    print("\n--- Computing Accuracy on Test Set ---")
    total = 0
    correct = 0
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size()[0]
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")


if __name__ == "__main__":
    sys.exit(int(main() or 0))
