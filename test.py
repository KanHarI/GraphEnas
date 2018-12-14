
import model

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import torch.optim as optim

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

supermodel = model.Supermodel()

SUBMODEL_LAYERS = 10
LAYERS_BETWEEN_HALVINGS = 4
OUTPUT_DIM = 10
SUBMODEL_CHANNELS = 20

sbm = supermodel.create_submodel(SUBMODEL_LAYERS, LAYERS_BETWEEN_HALVINGS, OUTPUT_DIM, SUBMODEL_CHANNELS)

if torch.cuda.is_available():
    sbm = sbm.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(sbm.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs
        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = sbm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1 == 0:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
