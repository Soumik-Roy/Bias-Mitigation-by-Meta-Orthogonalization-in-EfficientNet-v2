import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms as Trf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
from PIL import Image as PILImage
import trainTestFunctions

import efficientnetv2.effnetv2 as effnetv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clearPlots(fig=None):
    plt.figure().clear()
    if(fig!=None):
        fig.clear()
    plt.close('all')
    plt.cla()
    plt.clf()

dataset_transform = Trf.Compose([
    Trf.ToTensor(),
    Trf.Normalize((0.5), (0.5)),
    Trf.Resize((24, 24)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./datasets", 
    train=True, 
    transform=dataset_transform, 
    download=True,
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./datasets", 
    train=False, 
    transform=dataset_transform, 
    download=True,
)

output_classes = dict(train_dataset.class_to_idx)


label_map = {value:key for key, value in output_classes.items()}

print("Label map: ", label_map)
print()

n_channels = 3

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

images = None
labels = None
for (_images, _labels) in train_loader:
    images = _images
    labels = _labels
    break

def goodVisualizer(dataset=None, images=None, labels=None, saveFileName=None):
    '''A beautiful visualization of 10 images'''
    if dataset != None:
        dataiter = iter(dataset)
        images, labels = next(dataiter)
    
    figure = plt.figure(figsize=(12, 5))
    cols, rows = 5, 2
    for i in range(1, cols * rows + 1):
        img, label = images[i], labels[i].item()
        figure.add_subplot(rows, cols, i)
        plt.title(f"{label_map[label]} ({label})")
        plt.axis("off")
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0))*0.5 + 0.5)
    
    if saveFileName!=None:
        plt.savefig(saveFileName)
    # plt.show()
    clearPlots(figure)

goodVisualizer(images=images, labels=labels, saveFileName="dataset_visualization.png")


# train the model on CIFAR10 dataset using pytorch
learning_rate = 0.001
n_epochs = 10

model = effnetv2.effnetv2_s()  
model.to(device)

criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_stats1 = trainTestFunctions.train(
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    n_epochs,
)


plt.plot([x for x in range(1, len(train_stats1['losses'])+1)], train_stats1['losses'])
plt.title("Training Loss v/s Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("train_losses.png")
clearPlots()
# plt.show()

plt.plot([x for x in range(1, len(train_stats1['accuracies'])+1)], train_stats1['accuracies'])
plt.title("Training Accuracy v/s Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("train_accuracies.png")
clearPlots()
# plt.show()


test_stats1 = trainTestFunctions.test(model, test_loader, criterion)

print("Test Loss: {:.2f}\nTest Accuracy: {:.2f}".format(test_stats1['loss'], test_stats1['accuracy']))

# save the model
torch.save(model.state_dict(), "results/effnetv2_s_cifar10.pth")

# # load the model
# model.load_state_dict(torch.load("results/effnetv2_s_cifar10.pth"))

trainTestFunctions.plot_confusion_matrix(model, test_loader, list(output_classes.keys))
