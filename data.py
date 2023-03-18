import torch
import torchvision
from  torchvision import transforms


def get_data():

    transform = transforms.Compose([   
        transforms.ToTensor(), # convert image numpy array to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise the tensor (mean, stdev)
        ])

    trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, trainset, testset