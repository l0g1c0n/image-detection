import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from PIL import Image
# Data transformation and DataLoader
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = "fruits"
dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)

# Number of classes


# Create DataLoader for the dataset
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, num_workers=0)

# CNN Model
class Net(nn.Module):
    def __init__(self,num_channels = 3,width= 64,height =64,num_classes = 2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 16 * 64, 128)#self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Updated output layer for 3 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 16 * 64)#x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


input_channels = 3
input_height = 64
input_width = 64
num_classes = 10

model = Net(input_channels, input_height, input_width, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train(model, data_loader, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #if i % 100 == 99:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
    torch.save(model.state_dict(),"model.pth")


train(model, data_loader, criterion, optimizer, num_epochs=5)

import matplotlib.pyplot as plt
import numpy as np

# Load the trained model (assuming it's already trained)
#model = Net()  # Instantiate the model if needed
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load test images
test_data = datasets.ImageFolder(root=data_dir, transform=data_transform)
test_loader = DataLoader(test_data, batch_size=1,shuffle=True,  num_workers=0)

# Create a dictionary to map class indices to class labels
class_labels = dataset.classes

# Show some images along with their predictions
num_images_to_show = 8

num_rows = 2
num_cols = 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

for i, (image, label) in enumerate(test_loader, 1):
    if i > num_images_to_show:
        break

    output = model(image)
    _, predicted_class = torch.max(output, 1)
    predicted_label = class_labels[predicted_class.item()]
    true_label = class_labels[label.item()]

    # Convert tensor image to numpy array for visualization
    image_np = np.transpose(image.squeeze().numpy(), (1, 2, 0))
    image_np = image_np * 0.5 + 0.5  # Unnormalize the image

    # Calculate the row and column index for the current image
    row_idx = (i - 1) // num_cols
    col_idx = (i - 1) % num_cols

    axes[row_idx, col_idx].imshow(image_np)
    axes[row_idx, col_idx].set_title(f"Predicted: {predicted_label}, True: {true_label}")
    axes[row_idx, col_idx].axis("off")

# If there are fewer images than the grid size, hide empty subplots
for i in range(len(test_loader), num_rows * num_cols):
    row_idx = i // num_cols
    col_idx = i % num_cols
    axes[row_idx, col_idx].axis("off")

plt.tight_layout()  # Adjust the layout to avoid overlapping
plt.show()



#Function to predict the label for a given image
#def predict_label(image_path, model, class_labels):
#    image = Image.open(image_path)
#    image = data_transform(image).unsqueeze(0)  # Add batch dimension
#    output = model(image)
#    _, predicted_class = torch.max(output, 1)
#
#
#
#    predicted_label = class_labels[predicted_class.item()]
#    return predicted_label
#
#
#
#
#
#
#
## Input the image path from the user
#while True:
#    image_path = input("Enter the path of the image (or 'exit' to quit): ")
#    if image_path.lower() == "exit":
#        break
#
#    try:
#        predicted_label = predict_label(image_path, model, class_labels)
#        print(f"Predicted label: {predicted_label}")
#    except Exception as e:
#        print(f"Error: {e}")