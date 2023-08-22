import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import seaborn as sn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

"""
NOT USED
Ensemble model abandoned at this stage
very satisfied with the results from simple fine-tuning
"""

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'C:/Users/Lachie/Desktop/Star/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(class_names)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_accs = []
    val_accs = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_accs.append(torch.Tensor.cpu(epoch_acc).item())

            if phase == 'train':
                train_accs.append( torch.Tensor.cpu(epoch_acc).item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    plt.plot(np.linspace(1, 100, 100).astype(int), train_accs)
    plt.plot(np.linspace(1, 100, 100).astype(int), val_accs)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('model accuracy')
    plt.legend(['accuracy', 'validation accuracy'])
    plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.
# We aren't freezing the weights, so the convolutions should also be propogated back
model_resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
model_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# for param in model_ft.parameters():
#     param.requires_grad = False

n_inputs = model_resnet101.fc.in_features

model_resnet101.avgpool = nn.Sequential()
model_resnet101.fc = nn.Sequential(OrderedD)

model_resnet50.avgpool = nn.Sequential()
model_resnet50.fc = nn.Sequential(OrderedDict([

]))

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.resnet101 = models.resnet101(weight=models.ResNet101_Weights.IMAGENET1K_V1)



class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear()

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2))
        x = self.classifier(F.relu(x))
        return x

ensemble_model = 

num_ftrs = model_ft.fc.in_features


# make the fully connected layer the number of feature
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

# Cross Entropy loss
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, momentum=0.9)

step_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.2)

model = train_model(model_ft, criterion, optimizer_ft, step_lr_scheduler, num_epochs=100)

# visualize_model(model_ft)

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in dataloaders["val"]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

# constant for classes
classes = ('Arrive', 'Bed', 'Bird', 'Boy', 'Come', 'Day', 'Deer', 'Frog', 'Girl', 'Good', 'Lady', 'Laugh', 'Man', 'Night', 'People', 'Rabbit', 'Real', 'Same', 'Say', 'Sheep', 'Slow', 'Sprint', 'Think', 'Tortoise', 'What', 'Where', 'Window', 'Wolf', 'Yell')
# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('ResNet-101_confusion-matrix.png')