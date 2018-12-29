
import model

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import torch.optim as optim

import time
import math

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
if torch.cuda.is_available():
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
if torch.cuda.is_available():
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=True, num_workers=2)
else:
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=True)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

supermodel = model.Supermodel()

if torch.cuda.is_available():
    supermodel = supermodel.cuda()

train_weights_size = int(0.8 * len(trainset))
train_arch_size = len(trainset) - train_weights_size

weights_trainset, arch_trainset = torch.utils.data.random_split(trainset, [train_weights_size, train_arch_size])

def dataset_infigen(dataset):
    while True:
        for data in trainloader:
            yield data

SUBMODEL_LAYERS = 10
LAYERS_BETWEEN_HALVINGS = 4
OUTPUT_DIM = 10
SUBMODEL_CHANNELS = 10

sbm = supermodel.create_submodel(SUBMODEL_LAYERS, LAYERS_BETWEEN_HALVINGS, OUTPUT_DIM, SUBMODEL_CHANNELS)

if torch.cuda.is_available():
    sbm = sbm.cuda()

criterion = nn.CrossEntropyLoss()
weights_optimizer = optim.SGD([sbm.subgraph.parameters(), sbm.final_classifier.parameters()], lr=0.001, momentum=0.9)
actor_critic_optimizer = optim.Adam(sbm.supermodel.parameters())

PRINT_FREQUENCY = 100


weights_trainset = dataset_infigen(weights_trainset)
arch_trainset = dataset_infigen(arch_trainset)


TRAIN_STEP_TIME = 1.0 # Seconds
CRITIC_PLAN_LENGTH = 100

critic_preds = []
ground_truch_losses = []

last_loss = None

for i in range(10000):
    actor_critic_optimizer.zero_grad()
    actor_loss, critic_mean, critic_std = sbm.refresh_subgraph()
    print("actor_loss: ", actor_loss.item())
    critic_preds.append(critic_mean, critic_std)

    start_time = time.time()
    while (time.time() - start_time) < TRAIN_STEP_TIME:
        data = weights_trainset.__next__()
        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            weights_optimizer.zero_grad()

            outputs = sbm(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            weights_optimizer.step()

    data = arch_trainset.__next__()
    inputs, labels = data

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    outputs = sbm(inputs)
    loss = criterion(outputs, labels)
    print("Arch loss:", loss)
    if last_loss is None:
        last_loss = loss
    loss_delta = math.log(loss) - math.log(last_loss)

    ground_truch_losses.append(loss_delta)
    critic_loss = 0.0
    if len(ground_truch_losses == CRITIC_PLAN_LENGTH):
        it = 0.0
        for loss in ground_truch_losses[::-1]:
            it *= (1.0 - 1.0/(CRITIC_PLAN_LENGTH//2))
            it += loss
        loss = torch.tensor(loss)
        if torch.cuda.is_available():
            loss = loss.cuda()

        ground_truch_losses = ground_truch_losses[1:]
        critic_mean, critic_std = critic_preds[0]
        critic_preds = critic_preds[1:]

        # Calculate gaussian loss
        critic_loss = torch.pow(critic_std, -0.5) * torch.exp(-0.5 * (loss - critic_mean) * torch.pow(critic_std, -1))
        print("critic_loss:", critic_loss.item())

    actor_critic_loss = actor_loss + critic_loss
    actor_critic_loss.backward()
    actor_critic_optimizer.step()






for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs
        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        weights_optimizer.zero_grad()

        # forward + backward + optimize
        outputs = sbm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        weights_optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % PRINT_FREQUENCY == PRINT_FREQUENCY - 1:
            print('[%5d, %5d] loss: %f' %
                  (epoch + 1, i + 1, running_loss / PRINT_FREQUENCY))
            running_loss = 0.0
            print("Refreshing subgraph...")
            sbm.refresh_subgraph()
            print("Refreshed!")

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = sbm(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = sbm(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
