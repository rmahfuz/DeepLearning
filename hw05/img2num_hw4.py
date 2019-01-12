import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

#--------------------------------------------------------------------------------------
def findScore(idx, labels):
    tmp = (idx - labels).tolist()
    score = sum(list(map(lambda x: int(x==0), tmp)))
    for i in range(len(labels)):
        if labels[i] - idx[i] != 0:
            class_acc[labels[i]] -= 1   
    return score
#=========================================================================================
class img2num(nn.Module):

    def __init__(self):
        super(img2num, self).__init__()        
        self.fc1 = nn.Linear(784,300)
        self.fc2 = nn.Linear(300,10)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def train(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.MNIST(root='mnist/', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)

        criterion = nn.CrossEntropyLoss() #use NLL
        #criterion = nn.NLLLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(1):  # loop over the dataset multiple times
            #running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs.reshape(50,784))
                outputs = torch.tensor(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                '''if i%99 == 0:
                    print('just trained with 100 batches')'''
            print('finished an epoch ', epoch)
        print('Finished Training')
#=========================================================================================
net = img2num()
net.train()

transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.MNIST(root='mnist/', train=False,
                                               download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle= True)

classes = ('one', 'two', 'three', 'four',
           'five', 'six', 'seven', 'eight', 'nine', 'ten')

class_acc = [1000]*len(classes)
score = 0
for i, (inputs,labels) in enumerate(testloader, 0):
    out = net.forward(inputs.reshape(50,784))
    out = torch.tensor(out)
    vals, idx = out.max(1)
    score += findScore(idx, labels)
print('accuracy = ', score/10000)
print('Accuracy per class:')
for i in range(len(class_acc)):
    print(classes[i], ': ', class_acc[i]/1000)

assert sum(class_acc) == score
