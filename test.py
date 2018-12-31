
import model

import torch
import torch.nn.functional as F
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

SUBMODEL_LAYERS = 20
LAYERS_BETWEEN_HALVINGS = 4
OUTPUT_DIM = 10
SUBMODEL_CHANNELS = 20

sbm = supermodel.create_submodel(SUBMODEL_LAYERS, LAYERS_BETWEEN_HALVINGS, OUTPUT_DIM, SUBMODEL_CHANNELS)

if torch.cuda.is_available():
    sbm = sbm.cuda()

criterion = nn.CrossEntropyLoss()
weights_optimizer = optim.SGD(sbm.parameters(), lr=0.001, momentum=0.9)
actor_critic_optimizer = optim.SGD(sbm.supermodel.parameters(), lr=0.002, momentum=0.9)

PRINT_FREQUENCY = 20


weights_trainset = dataset_infigen(weights_trainset)
arch_trainset = dataset_infigen(arch_trainset)


TRAIN_STEP_TIME = 1.0 # Seconds
CRITIC_PLAN_LENGTH = 20

critic_preds = []
ground_truch_losses = []

last_loss = None

GAUSSIAN_FACTOR = (2*math.pi)**(-0.5)
GAMMA = (1.0 - 1.0/((1+CRITIC_PLAN_LENGTH)//2))

def print_if_verbose(v, *args):
    if v:
        print(*args)

for i in range(10000):
    verbose = (i%PRINT_FREQUENCY == 0)
    if i == 1000:
        # Start actor acting
        sbm.softmax.expt += 1.0
    print_if_verbose(verbose, "\n\nStart iteration: ", i)
    actor_critic_optimizer.zero_grad()
    actor_loss, na1, na2 = sbm.refresh_subgraph()
    print_if_verbose(verbose, "actor_loss: ", actor_loss.item())
    critic_preds.append((na1, na2))

    train_iter = 0

    start_time = time.time()
    while (time.time() - start_time) < TRAIN_STEP_TIME:
        train_iter += 1
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

    print_if_verbose(verbose, "weights train iterations:", train_iter)

    with torch.no_grad():
        correct = 0
        total = 0

        loss = 0.0

        for j in range((train_iter//5) + 1):
            data = arch_trainset.__next__()
            inputs, labels = data
        
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
        
            outputs = sbm(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
    
            loss += criterion(outputs, labels)

        loss = loss / ((train_iter//5) + 1)

        print_if_verbose(verbose, 'Accuracy of the network on the test batch images: %d %%' % (100 * correct / total))
        print_if_verbose(verbose, "Test batch loss:", loss.item())

    if last_loss is None:
        last_loss = loss
    loss_delta = loss - last_loss

    print_if_verbose(True, "loss_delta:", loss_delta.item())

    ground_truch_losses.append(loss_delta.item())

    critic_loss = 0.0
    if len(ground_truch_losses) == CRITIC_PLAN_LENGTH:
        it = 0.0
        for loss in ground_truch_losses[::-1]:
            it *= GAMMA
            it += loss
        loss = torch.tensor(it)
        if torch.cuda.is_available():
            loss = loss.cuda()

        ground_truch_losses = ground_truch_losses[1:]
        na1, na2 = critic_preds[0]
        critic_preds = critic_preds[1:]

        na1 = (sbm.supermodel.node_preprocessor(na1[0]), na1[1])
        na2 = (sbm.supermodel.node_preprocessor(na2[0]), na2[1])

        critic_res = sbm.supermodel.actor_critic_graphsage.forwardAB(na1, na2)
        critic_res = sbm.supermodel.critic(critic_res)

        print_if_verbose(verbose, "agg_loss:", loss.item())

        critic_mean = critic_res[0,0]
        print_if_verbose(verbose, "critic_mean:", critic_mean.item())
        critic_std = critic_res[0,1]
        
        # Softplus as std has to be positive
        critic_std = torch.log(1 + torch.exp(-torch.abs(critic_std))) + F.relu(critic_std)

        print_if_verbose(verbose, "critic_std:", critic_std.item())


        # Calculate gaussian loss
        critic_loss = -GAUSSIAN_FACTOR*torch.pow(critic_std, -0.5) * torch.exp(-0.5 * torch.pow((loss - critic_mean) * torch.pow(critic_std, -1), 2))
        # Add term for numerical stability - previously, it areas of relative 
        # stability, STD values dropped too low due to ADAM's momentum and
        # were stuck there with loss around 0.0
        critic_loss -= 1e-2 * torch.log(critic_std)


        print_if_verbose(verbose, "critic_loss:", critic_loss.item())

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
