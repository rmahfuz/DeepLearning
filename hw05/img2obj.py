import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import cv2


class img2obj(nn.Module):

    def __init__(self):
        super(img2obj, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5) #changed 1 to 3 because of 3 channels
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)
        self.classes = [
        'beaver', ' dolphin', ' otter', ' seal', ' whale', 
        'aquarium fish', ' flatfish', ' ray', ' shark', ' trout', 
        'orchids', ' poppies', ' roses', ' sunflowers', ' tulips', 
        'bottles', ' bowls', ' cans', ' cups', ' plates',
        'apples', ' mushrooms', ' oranges', ' pears', ' sweet peppers', 
        'clock', ' computer keyboard', ' lamp', ' telephone',' television', 
        'bed', ' chair', ' couch', ' table', ' wardrobe', 
        'bee', ' beetle', ' butterfly', ' caterpillar', ' cockroach', 
        'bear', ' leopard', ' lion', ' tiger', ' wolf', 
        'bridge', ' castle', ' house', ' road', ' skyscraper', 
        'cloud', ' forest', ' mountain', ' plain', ' sea', 
        'camel', ' cattle', ' chimpanzee', ' elephant', ' kangaroo', 
        'fox', ' porcupine', ' possum', ' raccoon', ' skunk', 
        'crab', ' lobster', ' snail', ' spider', ' worm',
        'baby', ' boy', ' girl', ' man', ' woman', 
        'crocodile', ' dinosaur', ' lizard', ' snake', ' turtle',
        'hamster', ' mouse', ' rabbit', ' shrew', ' squirrel', 
        'maple', ' oak', ' palm', ' pine', ' willow',
        'bicycle', ' bus', ' motorcycle', ' pickup truck', ' train', 
        'lawn-mower', ' rocket', ' streetcar', ' tank', ' tractor'
        ]

    def forward_to_train(self, x):
        #print(type(x), x.shape)
        # Max pooling over a (2, 2) window
        for i in range(x.shape[0]): #normalizing
            x[i] = x[i]/x[i].max()
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        '''x is a 3x32x32 tensor'''


        #print(type(x), x.shape)
        # Max pooling over a (2, 2) window
        x = x.type(torch.FloatTensor).view(1,3,32,32)
        x = x/x.max() #normalizing
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        vals, idx = x.max(1)
        #print('returning ', self.classes[idx])
        return self.classes[idx]
    
    
    def train(self):
        transform = transforms.Compose(
            [transforms.ToTensor()])#,
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)

        criterion = nn.CrossEntropyLoss() #use NLL
        #criterion = nn.NLLLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(0):  # loop over the dataset multiple times
            #running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                optimizer.zero_grad()
                outputs = self.forward_to_train(inputs)
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

    def view(self, x):
        '''x is a 3x32x32 byte tensor'''
        label = self.forward(x)
        cv2.imshow(label + '.png', x.numpy())
        print(label)
        
    def cam(self,idx=0):
        '''idx is an integer indicating the camera index'''
        camera = cv2.VideoCapture(idx)
        for i in range(10):
            return_value, image = camera.read()
            image = cv2.resize(image, (32,32))
            image = torch.from_numpy.view((3,32,32))
            self.view(image)
            #cv2.imwrite(path + 'opencv'+str(i)+'.png', image)
        del(camera)


net = img2obj()
net.train()

transform = transforms.Compose(
            [transforms.ToTensor()])#,
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle= True)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = [
'beaver', ' dolphin', ' otter', ' seal', ' whale', 
'aquarium fish', ' flatfish', ' ray', ' shark', ' trout', 
'orchids', ' poppies', ' roses', ' sunflowers', ' tulips', 
'bottles', ' bowls', ' cans', ' cups', ' plates',
'apples', ' mushrooms', ' oranges', ' pears', ' sweet peppers', 
'clock', ' computer keyboard', ' lamp', ' telephone',' television', 
'bed', ' chair', ' couch', ' table', ' wardrobe', 
'bee', ' beetle', ' butterfly', ' caterpillar', ' cockroach', 
'bear', ' leopard', ' lion', ' tiger', ' wolf', 
'bridge', ' castle', ' house', ' road', ' skyscraper', 
'cloud', ' forest', ' mountain', ' plain', ' sea', 
'camel', ' cattle', ' chimpanzee', ' elephant', ' kangaroo', 
'fox', ' porcupine', ' possum', ' raccoon', ' skunk', 
'crab', ' lobster', ' snail', ' spider', ' worm',
'baby', ' boy', ' girl', ' man', ' woman', 
'crocodile', ' dinosaur', ' lizard', ' snake', ' turtle',
'hamster', ' mouse', ' rabbit', ' shrew', ' squirrel', 
'maple', ' oak', ' palm', ' pine', ' willow',
'bicycle', ' bus', ' motorcycle', ' pickup truck', ' train', 
'lawn-mower', ' rocket', ' streetcar', ' tank', ' tractor'
]

class_acc = [100]*100

def findScore(idx, labels):
    tmp = (idx - labels).tolist()
    score = sum(list(map(lambda x: int(x==0), tmp)))
    for i in range(len(labels)):
        if labels[i] - idx[i] != 0:
            class_acc[labels[i]] -= 1   
    return score

score = 0
for i, (inputs,labels) in enumerate(testloader, 0):
    out = net.forward(inputs.view((3,32,32)));
    idx = classes.index(out)
    #vals, idx = out.max(1)
    score += findScore(torch.tensor([idx]), labels)
    #net.view(inputs.view((3,32,32)))
    
print('accuracy = ', score/10000)
print('Accuracy per class:')
for i in range(len(class_acc)):
    print(classes[i], ': ', class_acc[i]/100)

assert sum(class_acc) == score
