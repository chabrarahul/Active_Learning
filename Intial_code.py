import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats 
import csv
import imageio
import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

batch_size = 25
k =10 #total number of class
gradient = np.zeros(100000, dtype=float).reshape(10,10000) #Matrix of gradient. row = classes and columns = image
c_i = np.zeros(100000, dtype=float).reshape(10,10000) #Matrix of gradient. row = classes and columns = image
class_order = np.arange(100000).reshape(10,10000) #Matrix of gradient. row = classes and columns = image


# function will return indices list for a batch 
def batch_label(output, k):
    sorting, indices = torch.sort(output, descending=True) # sorting stores the sorting probabolities and indices stores the class in descending order 
    indices = indices.cpu().numpy() # converting tensor array to numpy array  
    indices = indices[:,:k] # slicing the indices
    indices = np.transpose(indices) # indices dimension = column = image and row = indices in descending order
    return indices

# Stores the gradient, probablity and class order in matrix. 
# grad = gradient of complete batch, output = model output, image_index = list of index of images, count = keeps the record of class 
def complete_dict(grad, output,image_index, count, batch_size):
    sorting, indices = torch.sort(output, descending=True)    
    image_index = image_index.cpu().numpy()
    for batch_number in range(batch_size): 
        mul = torch.norm(grad[batch_number], 2) # calculates the norm of gradient of image with perticular class
        ind = image_index[batch_number] # index of the image 
        gradient[count][ind] = mul.detach().cpu().numpy()  # storing the value in gradient matrix        
        c_i[count][ind] = sorting[batch_number][count].detach().cpu().numpy() # Storing the value probability in matrix        
        class_order[count][ind] = indices[batch_number][count] # Storing the class in matrix 


# Loading the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root = './data/CIFAR10', train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root = './data/CIFAR10', train=False, transform = transforms.ToTensor(), download=True)

# Adding index to dataset
train_set_data = []
count = 0
for image in train_set:
    random = (image[0], image[1]) + (count,) 
    train_set_data.append(random)
    count = count + 1
    
test_set_data = []

num = 0
for image in test_set:
    random = (image[0], image[1]) + (num,) 
    test_set_data.append(random)
    num = num + 1

#Custom data loader with image index
class Data(torch.utils.data.Dataset):
    def __init__(self, train_or_test,transform=None):
                super(Data, self).__init__()
                if train_or_test == 'train':
                        self.dataset = train_set_data
                        self.length = len(train_set_data)
                if train_or_test == 'test':
                        self.dataset = test_set_data
                        self.length = len(test_set_data)
    def __len__(self):
                return self.length
    
    def __getitem__(self, idx):
                image = self.dataset[idx][0] 
                label = self.dataset[idx][1] 
                index = self.dataset[idx][2] 
                sample = {'image' : image, 
                          'label' : label,
                          'index' : index}
        
                #if self.transform:
                #     sample = self.transform(sample)
                return sample

train_data = Data(train_or_test = "train")
test_data = Data(train_or_test = "test")

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle= True, sampler=None,batch_sampler=None, num_workers=0, collate_fn=None,pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size= 25, shuffle= False, sampler=None,batch_sampler=None, num_workers=0, collate_fn=None,pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None)


dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model 
class TemplateNet(nn.Module):
    def __init__(self):
        super(TemplateNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(28800, 1000)
        #self.fc1 = nn.Linear(6272, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))        
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 28800)
        #x = x.view(-1, 6272)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

net = TemplateNet()

# Training 
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, labels, image_index = data['image'],data['label'], data['index']
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 12000 == 11999:
            print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(12000)))
            running_loss = 0.0

# Testing 
for i, data in enumerate(test_data_loader):
    inputs, labels, image_index = data['image'],data['label'], data['index']
    inputs = inputs.to(device)
    labels = labels.to(device)
    inputs.requires_grad = True
    optimizer.zero_grad()
    image_index = image_index.to(device)
    output = net(inputs)
    indices = batch_label(output,k)
    counts = 0
    for index in indices:
        net.zero_grad()
        index = torch.from_numpy(index).to(device)
        loss = criterion(output, index)
        grad_input = torch.autograd.grad(loss, inputs,retain_graph = True, allow_unused=True)# Calculating gradient of complete batch
        complete_dict(grad_input[0], output, image_index, counts, batch_size)
        counts = counts+1
    

print(class_order)
print(gradient)
print(c_i)
