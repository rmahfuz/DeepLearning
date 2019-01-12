import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

def convert(x):
    '''converts 50 x 1 x 28 x 28 MNIST data into 50 x 1 x 32 x 32'''
    tmp = torch.zeros((50,1,28,4))
    x = torch.cat((x,tmp),3)
    tmp = torch.zeros((50,1,4,32))
    x = torch.cat((x,tmp),2)
    #print('x.shape = ', x.shape) #should be (50,1,32,32)
    return x
    
class img2num(nn.Module):

    def __init__(self):
        super(img2num, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5) #changed 1 to 3 because of 3 channels
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.classes = ('one', 'two', 'three', 'four',
           'five', 'six', 'seven', 'eight', 'nine', 'ten')


    def forward_to_train(self, x):
        x = x/x.max() #normalizing
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def forward(self, x):
        '''x is a 28 x 28 tensor'''
        x = inputs.view((28,28))
        tmp = torch.zeros((28,4))
        x = torch.cat((x,tmp),1)
        tmp = torch.zeros((4,32))
        x = torch.cat((x,tmp),0)

        x = x.type(torch.FloatTensor).view(1,1,32,32)
        x = x/x.max() #normalizing
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        vals, idx = x.max(1)
        return self.classes[idx]
    
    def train(self):
        transform = transforms.Compose(
            [transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='mnist/', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)

        criterion = nn.CrossEntropyLoss() #use NLL
        #criterion = nn.NLLLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(1):  # loop over the dataset multiple times
            #running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                #print('labels = ', labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                #outputs = self.forward(inputs)
                #print(inputs.shape)
                outputs = self.forward_to_train(convert(inputs))
                #print('outputs = ', outputs)
                '''labels_new = torch.zeros((4,10), dtype = torch.long)
                for j in range(4):
                    labels_new[j,labels[j]] = 1'''
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print('finished an epoch')

        print('Finished Training')


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = img2num()
net.train()

transform = transforms.Compose(
            [transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='mnist/', train=False,
                                               download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle= True)

classes = ('one', 'two', 'three', 'four',
           'five', 'six', 'seven', 'eight', 'nine', 'ten')

class_acc = [1000]*len(classes)

def findScore(idx, labels):
    tmp = (idx - labels).tolist()
    score = sum(list(map(lambda x: int(x==0), tmp)))
    for i in range(len(labels)):
        if labels[i] - idx[i] != 0:
            class_acc[labels[i]] -= 1   
    return score

score = 0
for i, (inputs,labels) in enumerate(testloader, 0):
    '''x = inputs.view((28,28))
    tmp = torch.zeros((28,4))
    x = torch.cat((x,tmp),1)
    tmp = torch.zeros((4,32))
    inputs = torch.cat((x,tmp),0)
    #print(inputs.shape)'''
    
    out = net.forward(inputs.view((1,28,28)))
    #vals, idx = out.max(1)
    idx = classes.index(out)
    score += findScore(torch.tensor([idx]), labels)
print('accuracy = ', score/10000)
print('Accuracy per class:')
for i in range(len(class_acc)):
    print(classes[i], ': ', class_acc[i]/1000)

assert sum(class_acc) == score
